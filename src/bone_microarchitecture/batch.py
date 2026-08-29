"""Manifest-driven batch workflow for microarchitecture measurements."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from bone_imaging_derivatives import (
    DerivativeManifest,
    DerivativeProgressEvent,
    DerivativeRecord,
    discover_manifests,
    format_progress_event,
    write_manifest,
)
from bone_imaging_derivatives.layout import manifest_path, record_output_path

from .pipeline import compute_microarchitecture
from .results import write_measurement_csv


_REQUIRED_ROLES = ("bone_segmentation", "periosteal_mask", "trabecular_mask")
_FALLBACK_NAMES = {
    "transformed_image": ("image.npy", "grayscale.npy"),
    "bone_segmentation": ("bone_mask.npy", "bone.npy"),
    "periosteal_mask": ("periosteal_mask.npy", "periosteal.npy"),
    "trabecular_mask": ("trabecular_mask.npy", "trabecular.npy"),
    "cortical_mask": ("cortical_mask.npy", "cortical.npy"),
    "scan_region_native_common": ("common_region.npy", "common_region_mask.npy"),
}


def run_microarchitecture_batch(
    dataset_root,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_common_region: bool = True,
    thickness_method: str = "hildebrand",
    thickness_backend: str = "auto",
    progress: Callable[[DerivativeProgressEvent], None] | None = None,
) -> list[DerivativeRecord]:
    """Measure every manifest-discovered case and write Microarchitecture outputs.

    The workflow intentionally only owns derivative discovery and output writing.
    It clips biological masks to the optional common scan-region mask before
    calling :func:`compute_microarchitecture`, which retains the one-case API.
    Simple, unambiguous ``.npy`` filenames are accepted when manifests are not
    yet available, primarily for lightweight command-line workflows.
    """
    root = Path(dataset_root).resolve()
    cases = _discover_cases(root)
    if not cases:
        raise ValueError("No complete microarchitecture input case was found")

    output_records: list[DerivativeRecord] = []
    for case in cases:
        subject_id, site, session_id, stack_index, space = _case_key(case["bone_segmentation"])
        _emit(progress, subject_id, site, session_id, "measure", "started", "Computing microarchitecture")
        masks = {role: _load_array(record.path) for role, record in case.items() if role != "transformed_image"}
        image = _load_array(case["transformed_image"].path)
        common = masks.pop("scan_region_native_common", None) if use_common_region else None
        if common is not None:
            common = np.asarray(common) > 0
            masks = {role: (np.asarray(mask) > 0) & common for role, mask in masks.items()}

        result = compute_microarchitecture(
            grayscale=image,
            bone_mask=masks["bone_segmentation"],
            periosteal_mask=masks["periosteal_mask"],
            trabecular_mask=masks["trabecular_mask"],
            cortical_mask=masks.get("cortical_mask"),
            spacing=spacing,
            thickness_method=thickness_method,
            thickness_backend=thickness_backend,
        )
        session_part = f"ses-{session_id}" if session_id else "ses-unknown"
        filename = f"sub-{subject_id}_{session_part}_site-{site}_measurements.csv"
        output_path = record_output_path(
            root, "Microarchitecture", subject_id, site, session_part, "tables", filename
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_measurement_csv(output_path, result.measurements, result.maps)
        output_records.append(
            DerivativeRecord(
                derivative="Microarchitecture",
                role="measurements_table",
                subject_id=subject_id,
                site=site,
                session_id=session_id,
                stack_index=stack_index,
                space="table",
                path=output_path,
                source="generated",
                inputs=tuple(record.record_id for record in case.values()),
                metadata={
                    "use_common_region": common is not None,
                    "thickness_method": thickness_method,
                    "thickness_backend": result.metadata["thickness_backend"],
                },
                content_type="table",
            )
        )
        _emit(progress, subject_id, site, session_id, "measure", "completed", "Wrote measurements", output_path)

    manifest = DerivativeManifest.create(
        "Microarchitecture", root, {"name": "bone-microarchitecture", "version": "0.1.0"},
        records=tuple(output_records),
    )
    write_manifest(manifest, manifest_path(root, "Microarchitecture"))
    return output_records


def _discover_cases(root: Path) -> list[dict[str, DerivativeRecord]]:
    records = [record for manifest in discover_manifests(root) for record in manifest.records]
    cases: list[dict[str, DerivativeRecord]] = []
    for bone in (record for record in records if record.role == "bone_segmentation"):
        key = _case_key(bone)
        case = {"bone_segmentation": bone}
        for role in ("periosteal_mask", "trabecular_mask", "cortical_mask", "scan_region_native_common"):
            match = _matching_record(records, role, key)
            if match is not None:
                case[role] = match
        image = _matching_record(records, "transformed_image", key)
        if image is None:
            image = _matching_image_record(records, key)
        if image is not None:
            case["transformed_image"] = image
        if all(role in case for role in (*_REQUIRED_ROLES, "transformed_image")):
            cases.append(case)
    return cases or _fallback_case(root)


def _matching_record(records, role: str, key):
    return next((record for record in records if record.role == role and _case_key(record) == key), None)


def _matching_image_record(records, key):
    return next((record for record in records if record.content_type == "image" and _case_key(record) == key), None)


def _fallback_case(root: Path) -> list[dict[str, DerivativeRecord]]:
    paths = {role: _find_fallback_path(root, names) for role, names in _FALLBACK_NAMES.items()}
    if any(paths[role] is None for role in (*_REQUIRED_ROLES, "transformed_image")):
        return []
    return [{
        role: DerivativeRecord(
            derivative="Segmentation" if role != "transformed_image" else "Registration",
            role=role,
            subject_id="unknown",
            site="unknown",
            session_id=None,
            stack_index=None,
            space="native",
            path=path,
            source="provided",
            content_type="image" if role == "transformed_image" else "mask",
        )
        for role, path in paths.items() if path is not None
    }]


def _find_fallback_path(root: Path, names: tuple[str, ...]) -> Path | None:
    found = [path for name in names for path in root.rglob(name)]
    return found[0] if len(found) == 1 else None


def _case_key(record: DerivativeRecord) -> tuple[str, str, str | None, int | None, str]:
    return record.subject_id, record.site, record.session_id, record.stack_index, record.space


def _load_array(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"Only .npy inputs are currently supported: {path}")
    return np.load(path, allow_pickle=False)


def _emit(progress, subject_id, site, session_id, step, status, message, path=None) -> None:
    event = DerivativeProgressEvent("Microarchitecture", subject_id, site, session_id, step, status, message, path)
    if progress is None:
        print(format_progress_event(event))
    else:
        progress(event)

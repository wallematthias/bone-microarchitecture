"""Manifest-driven batch workflow for microarchitecture measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from bone_imaging_derivatives import (
    DerivativeManifest,
    DerivativeProgressEvent,
    DerivativeRecord,
    discover_artifacts,
    discover_manifests,
    format_progress_event,
    normalize_session_id,
    normalize_site,
    read_manifest,
    write_manifest,
)
from bone_imaging_derivatives.layout import manifest_path, record_output_path, voi_token

from .pipeline import compute_microarchitecture
from .metrics import compartment_metrics, masked_mean_sd
from .results import write_measurement_csv
from .thickness import summary


_REQUIRED_ROLES = ("bone_segmentation", "periosteal_mask", "trabecular_mask")
_FALLBACK_NAMES = {
    "transformed_image": ("image.npy", "grayscale.npy"),
    "bone_segmentation": ("bone_mask.npy", "bone.npy"),
    "periosteal_mask": ("periosteal_mask.npy", "periosteal.npy"),
    "trabecular_mask": ("trabecular_mask.npy", "trabecular.npy"),
    "cortical_mask": ("cortical_mask.npy", "cortical.npy"),
    "scan_region_native_common": ("common_region.npy", "common_region_mask.npy"),
}
_MAP_ROLES = {
    "Tb.Th": "trabecular_thickness_map",
    "Tb.Sp": "trabecular_spacing_map",
    "Tb.N": "trabecular_number_map",
    "Ct.Th": "cortical_thickness_map",
    "Ct.Po.Dm": "cortical_porosity_map",
    "Tt.BMD": "material_map",
    "Tb.BMD": "material_map",
    "Ct.BMD": "material_map",
}
_DISCOVERY_ROLE_MAP = {
    "image": "transformed_image",
    "segmentation": "bone_segmentation",
    "bone_segmentation": "bone_segmentation",
    "full": "periosteal_mask",
    "periosteal_mask": "periosteal_mask",
    "trab": "trabecular_mask",
    "trabecular_mask": "trabecular_mask",
    "cort": "cortical_mask",
    "cortical_mask": "cortical_mask",
    "scan_region_native_common": "scan_region_native_common",
    "common_region": "scan_region_native_common",
}


@dataclass(frozen=True)
class _LoadedVolume:
    array: np.ndarray
    spacing: tuple[float, float, float] | None = None
    sitk_image: object | None = None


def run_microarchitecture_batch(
    dataset_root,
    *,
    spacing: tuple[float, float, float] | None = None,
    use_common_region: bool = True,
    force: bool = False,
    require_common_region: bool = False,
    thickness_method: str = "hildebrand",
    thickness_backend: str = "auto",
    subject_id: str = "",
    site: str = "",
    session_id: str = "",
    progress: Callable[[DerivativeProgressEvent], None] | None = None,
) -> list[DerivativeRecord]:
    """Measure every manifest-discovered case and write Microarchitecture outputs.

    The workflow intentionally only owns derivative discovery and output writing.
    It clips biological masks to the optional common scan-region mask before
    calling :func:`compute_microarchitecture`, which retains the one-case API.
    Simple, unambiguous ``.npy`` filenames are accepted when manifests are not
    yet available, primarily for lightweight command-line workflows.
    """
    root = _dataset_root(Path(dataset_root).resolve())
    cases = _filter_cases(
        _discover_case_rows(root),
        subject_id=subject_id,
        site=site,
        session_id=session_id,
    )
    incomplete = [case for case in cases if _missing_required_roles(case, require_common_region=require_common_region)]
    if incomplete:
        raise ValueError(_incomplete_case_message(incomplete, require_common_region=require_common_region))
    if not cases:
        raise ValueError("No complete microarchitecture input case was found")

    existing_records = _read_existing_records(root)
    output_records: list[DerivativeRecord] = []
    measurement_records: list[DerivativeRecord] = []
    for case in cases:
        subject_id, site, session_id, stack_index, space = _case_key(case["bone_segmentation"])
        image = _load_record_volume(case["transformed_image"])
        case_spacing = image.spacing if image.spacing is not None else spacing
        if case_spacing is None:
            raise ValueError("spacing must be supplied for .npy batch inputs without image geometry")
        input_roles = ["transformed_image", "bone_segmentation", "periosteal_mask", "trabecular_mask"]
        if "cortical_mask" in case:
            input_roles.append("cortical_mask")
        if use_common_region and "scan_region_native_common" in case:
            input_roles.append("scan_region_native_common")
        input_ids = tuple(case[role].record_id for role in input_roles)
        native_input_roles = ["transformed_image", "bone_segmentation", "periosteal_mask", "trabecular_mask"]
        if "cortical_mask" in case:
            native_input_roles.append("cortical_mask")
        native_input_ids = tuple(case[role].record_id for role in native_input_roles)
        settings_hash = _compatibility_hash(input_ids, case_spacing, use_common_region, thickness_method, thickness_backend)
        native_map_hash = _compatibility_hash(native_input_ids, case_spacing, False, thickness_method, thickness_backend)
        case_key = _output_case_key(case["bone_segmentation"])
        native_map_records = None if force else _find_compatible_native_map_records(
            existing_records, case_key, native_input_ids, native_map_hash, case
        )
        reused = None if force or native_map_records is None else _find_compatible_measurement_record(
            existing_records, case_key, input_ids, settings_hash, case
        )
        if reused is not None:
            measurement_records.append(reused)
            _emit(progress, subject_id, site, session_id, "measure", "reused", "Reused compatible measurements", reused.path)
            continue

        masks = {role: _load_record_volume(record).array for role, record in case.items() if role != "transformed_image"}
        common = masks.pop("scan_region_native_common", None) if use_common_region else None
        common_mask = np.asarray(common) > 0 if common is not None else None

        native_maps = None if native_map_records is None else _load_native_maps_from_records(native_map_records)
        if native_maps is None:
            native_map_records = None
            _emit(progress, subject_id, site, session_id, "measure", "started", "Computing microarchitecture")
            result = compute_microarchitecture(
                grayscale=image.array,
                bone_mask=masks["bone_segmentation"],
                periosteal_mask=masks["periosteal_mask"],
                trabecular_mask=masks["trabecular_mask"],
                cortical_mask=masks.get("cortical_mask"),
                spacing=case_spacing,
                thickness_method=thickness_method,
                thickness_backend=thickness_backend,
            )
            native_measurements = result.measurements
            native_maps = result.maps
            resolved_backend = result.metadata["thickness_backend"]
        else:
            map_dir = next(iter(native_map_records.values())).path.parent if native_map_records else None
            _emit(
                progress,
                subject_id,
                site,
                session_id,
                "maps",
                "reused",
                "Reused native microarchitecture maps",
                map_dir,
            )
            _emit(progress, subject_id, site, session_id, "measure", "started", "Summarizing existing microarchitecture maps")
            native_measurements = _summarize_measurements(
                image.array,
                masks,
                native_maps,
                case_spacing,
            )
            resolved_backend = thickness_backend
        measurement_masks = masks
        if common_mask is not None:
            measurement_masks = {role: (np.asarray(mask) > 0) & common_mask for role, mask in masks.items()}
        measurements = native_measurements if common_mask is None else _summarize_measurements(
            image.array,
            measurement_masks,
            native_maps,
            case_spacing,
        )
        measurement_maps = _masked_maps_for_measurements(native_maps, measurement_masks)
        session_part = f"ses-{session_id}" if session_id else "ses-unknown"
        measurement_dir = "registered_measurements" if common_mask is not None else "measurements"
        filename = f"sub-{subject_id}_{session_part}_voi-{voi_token(site)}_measurements.csv"
        output_path = record_output_path(
            root, "Microarchitecture", subject_id, site, session_part, measurement_dir, filename
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_measurement_csv(output_path, measurements, measurement_maps)
        record_metadata = {
            "use_common_region": common_mask is not None,
            "thickness_method": thickness_method,
            "thickness_backend": resolved_backend,
            "settings_hash": settings_hash,
        }
        measurement_record = DerivativeRecord(
                derivative="Microarchitecture",
                role="measurements_table",
                subject_id=subject_id,
                site=site,
                session_id=session_id,
                stack_index=stack_index,
                space="table",
                path=output_path,
                source="generated",
                inputs=input_ids,
                metadata=record_metadata,
                content_type="table",
                settings_hash=settings_hash,
        )
        output_records.append(measurement_record)
        measurement_records.append(measurement_record)
        if native_map_records is None:
            for map_name, map_array in native_maps.items():
                role = _MAP_ROLES.get(map_name, "material_map")
                extension = ".nii.gz" if image.sitk_image is not None else ".npy"
                map_filename = f"sub-{subject_id}_{session_part}_voi-{voi_token(site)}_map-{map_name.lower().replace('.', '-')}{extension}"
                map_path = record_output_path(
                    root, "Microarchitecture", subject_id, site, session_part, "maps", map_filename
                )
                _write_map(map_path, map_array, image)
                output_records.append(
                    DerivativeRecord(
                        derivative="Microarchitecture",
                        role=role,
                        subject_id=subject_id,
                        site=site,
                        session_id=session_id,
                        stack_index=stack_index,
                        space="native",
                        path=map_path,
                        source="generated",
                        inputs=native_input_ids,
                        metadata={
                            "use_common_region": False,
                            "thickness_method": thickness_method,
                            "thickness_backend": resolved_backend,
                            "settings_hash": native_map_hash,
                            "map_name": map_name,
                        },
                        content_type="image",
                        settings_hash=native_map_hash,
                    )
                )
        _emit(progress, subject_id, site, session_id, "measure", "completed", "Wrote measurements", output_path)

    if output_records:
        superseded = {(_output_case_key(record), _record_replacement_signature(record)) for record in output_records}
        merged_records = [
            record for record in existing_records
            if (_output_case_key(record), _record_replacement_signature(record)) not in superseded
        ]
        merged_records.extend(output_records)
        manifest = DerivativeManifest.create(
            "Microarchitecture", root, {"name": "bone-microarchitecture", "version": "0.2.3"},
            records=tuple(merged_records),
        )
        write_manifest(manifest, manifest_path(root, "Microarchitecture"))
    return measurement_records


def _filter_cases(cases, *, subject_id: str = "", site: str = "", session_id: str = ""):
    subject_id = str(subject_id or "").strip()
    site = _filter_site(site)
    session_id = str(session_id or "").strip()
    if not subject_id and not site and not session_id:
        return list(cases)
    filtered = []
    for case in cases:
        case_subject, case_site, case_session, _stack_index, _space = _case_key_from_case(case)
        if subject_id and str(case_subject) != subject_id:
            continue
        if site and not _sites_match(case_site, site):
            continue
        if session_id and _session_key(case_session) != _session_key(session_id):
            continue
        filtered.append(case)
    return filtered


def _dataset_root(root: Path) -> Path:
    return root.parent if root.name == "derivatives" else root


def _discover_cases(root: Path) -> list[dict[str, DerivativeRecord]]:
    return [case for case in _discover_case_rows(root) if not _missing_required_roles(case)]


def _discover_case_rows(root: Path) -> list[dict[str, DerivativeRecord]]:
    records = [record for manifest in discover_manifests(root) for record in manifest.records]
    records.extend(_discover_shared_artifact_records(root))
    records.extend(_discover_structured_input_records(root))
    records = list(_deduplicate_records(sorted(records, key=lambda record: _record_source_priority(root, record))))
    cases: list[dict[str, DerivativeRecord]] = []
    keys = {
        _case_key(record)
        for record in records
        if record.role in {"bone_segmentation", "transformed_image", "source_image_view"}
    }
    for key in sorted(keys, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        case: dict[str, DerivativeRecord] = {}
        bone = _matching_record(records, "bone_segmentation", key)
        if bone is not None:
            case["bone_segmentation"] = bone
        for role in ("periosteal_mask", "trabecular_mask", "cortical_mask", "scan_region_native_common"):
            match = _matching_record(records, role, key)
            if match is not None:
                case[role] = match
        image = _matching_record(records, "transformed_image", key)
        if image is None:
            image = _matching_image_record(records, key)
        if image is not None:
            case["transformed_image"] = image
        if case:
            cases.append(case)
    return cases or _fallback_case(root)


def _record_source_priority(root: Path, record: DerivativeRecord) -> int:
    try:
        parts = [part.lower().replace("-", "") for part in record.path.resolve().relative_to(root).parts]
    except ValueError:
        parts = []
    family = ""
    if "derivatives" in parts:
        index = parts.index("derivatives")
        if index + 1 < len(parts):
            family = parts[index + 1]
    if not family:
        family = record.derivative.lower().replace("-", "")
    return {"importedcontours": 0, "bonecontours": 1}.get(family, 2)


def _missing_required_roles(case: dict[str, DerivativeRecord], *, require_common_region: bool = False) -> tuple[str, ...]:
    roles = [*_REQUIRED_ROLES, "transformed_image"]
    if require_common_region:
        roles.append("scan_region_native_common")
    return tuple(role for role in roles if role not in case)


def _case_key_from_case(case: dict[str, DerivativeRecord]) -> tuple[str, str, str | None, int | None, str]:
    record = case.get("bone_segmentation") or case.get("transformed_image")
    if record is None:
        raise ValueError("Microarchitecture case has no identifiable input artifact")
    return _case_key(record)


def _incomplete_case_message(cases: list[dict[str, DerivativeRecord]], *, require_common_region: bool = False) -> str:
    descriptions = []
    for case in cases:
        subject_id, site, session_id, stack_index, _space = _case_key_from_case(case)
        identity = f"sub-{subject_id}, ses-{session_id or 'unknown'}, voi-{site}"
        if stack_index is not None:
            identity = f"{identity}, stack-{stack_index:02d}"
        descriptions.append(
            f"{identity}: missing {', '.join(_missing_required_roles(case, require_common_region=require_common_region))}"
        )
    return "Microarchitecture batch prerequisites are incomplete: " + "; ".join(descriptions)


def _discover_shared_artifact_records(root: Path) -> list[DerivativeRecord]:
    """Convert shared artifact discovery records into microarchitecture inputs."""
    index = discover_artifacts(root, include_derivatives=True)
    records: list[DerivativeRecord] = []
    for artifact in index.records:
        if not _artifact_is_microarchitecture_input_source(root, artifact.path):
            continue
        role = _DISCOVERY_ROLE_MAP.get(artifact.role)
        if role is None:
            continue
        if role == "transformed_image" and artifact.kind != "image":
            continue
        if role != "transformed_image" and artifact.kind != "mask":
            continue
        if not artifact.subject_id or not artifact.site:
            continue
        derivative = "Registration" if role == "transformed_image" else "Segmentation"
        records.append(
            DerivativeRecord(
                derivative=derivative,
                role=role,
                subject_id=artifact.subject_id,
                site=artifact.site,
                session_id=artifact.session_id,
                stack_index=artifact.stack_index,
                space="native",
                path=artifact.path,
                source="provided",
                content_type="image" if role == "transformed_image" else "mask",
                metadata={
                    "discovery": {
                        "format": artifact.format,
                        "identity_confidence": artifact.identity_confidence,
                        "subject_source": artifact.subject_source,
                        "session_source": artifact.session_source,
                        "site_source": artifact.site_source,
                        "role_source": artifact.role_source,
                    }
                },
            )
        )
    return records


def _artifact_is_microarchitecture_input_source(root: Path, path: Path) -> bool:
    try:
        parts = [part.lower().replace("-", "_") for part in Path(path).resolve().relative_to(root).parts]
    except ValueError:
        return True
    if "derivatives" not in parts:
        return True
    index = parts.index("derivatives")
    if index + 1 >= len(parts):
        return False
    family = parts[index + 1]
    return family in {
        "importedcontours",
        "iplcontours",
        "bonecontours",
        "segmentation",
        "registration",
        "calibration",
        "commonregion",
        "common_region",
    }


def _deduplicate_records(records):
    seen = set()
    for record in records:
        key = (
            record.role,
            record.subject_id,
            record.site,
            record.session_id,
            record.stack_index,
            record.space,
        )
        if key in seen:
            continue
        seen.add(key)
        yield record


def _discover_structured_input_records(root: Path) -> list[DerivativeRecord]:
    """Discover raw scans and masks from lightweight on-disk conventions.

    This complements manifest discovery for two common cases: Scanco masks kept
    beside the source image, and Bone Contouring outputs under
    ``derivatives/Segmentation`` before a shared manifest has been written.
    """
    records: list[DerivativeRecord] = []
    for path in sorted(path for path in root.iterdir() if path.is_file() and _is_supported_volume(path)):
        parsed = _parse_structured_path(root, path)
        if parsed is None:
            continue
        role, subject_id, site, session_id, stack_index = parsed
        derivative = "Segmentation" if role != "transformed_image" else "Registration"
        records.append(
            _structured_record(root, path, derivative, role, subject_id, site, session_id, stack_index)
        )

    segmentation_root = root / "derivatives" / "Segmentation"
    if segmentation_root.exists():
        for path in sorted(path for path in segmentation_root.rglob("*") if path.is_file() and _is_supported_volume(path)):
            parsed = _parse_structured_path(root, path)
            if parsed is None:
                continue
            role, subject_id, site, session_id, stack_index = parsed
            if role == "transformed_image":
                continue
            records.append(
                _structured_record(root, path, "Segmentation", role, subject_id, site, session_id, stack_index)
            )
    return records


def _structured_record(
    root: Path,
    path: Path,
    derivative: str,
    role: str,
    subject_id: str,
    site: str,
    session_id: str | None,
    stack_index: int | None,
) -> DerivativeRecord:
    return DerivativeRecord(
        derivative=derivative,
        role=role,
        subject_id=subject_id,
        site=site,
        session_id=session_id,
        stack_index=stack_index,
        space="native",
        path=path,
        source="provided",
        content_type="image" if role == "transformed_image" else "mask",
    )


def _parse_structured_path(root: Path, path: Path) -> tuple[str, str, str, str | None, int | None] | None:
    role = _role_from_filename(path)
    if role is None:
        return None
    subject_id, site, session_id, stack_index = _metadata_from_path_parts(root, path)
    filename_metadata = _metadata_from_filename(path)
    subject_id = subject_id or filename_metadata.get("subject_id", "")
    filename_site = filename_metadata.get("site", "")
    if filename_site and (_is_generic_site(site) or not site):
        site = filename_site
    else:
        site = site or filename_site
    session_id = session_id or filename_metadata.get("session_id")
    if not subject_id or not site:
        return None
    return role, subject_id, site, session_id, stack_index


def _metadata_from_path_parts(root: Path, path: Path) -> tuple[str, str, str | None, int | None]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    subject_id = ""
    site = ""
    session_id = None
    stack_index = None
    for part in parts:
        if part.startswith("sub-"):
            subject_id = part[4:]
        elif part.startswith("site-"):
            site = _canonical_site(part[5:])
        elif part.startswith("voi-"):
            site = _canonical_site(part[4:])
        elif part.startswith("ses-"):
            session_id = normalize_session_id(part[4:])
        elif part.startswith("stack-"):
            try:
                stack_index = int(part[6:])
            except ValueError:
                stack_index = None
    return subject_id, site, session_id, stack_index


def _metadata_from_filename(path: Path) -> dict[str, str]:
    stem = _volume_stem(path)
    metadata: dict[str, str] = {}
    bids = re.search(r"(?:^|_)sub-([^_]+).*?(?:^|_)ses-([^_]+).*?(?:^|_)(?:site|voi)-([^_]+)", stem)
    if bids:
        metadata["subject_id"] = bids.group(1)
        metadata["session_id"] = normalize_session_id(bids.group(2))
        metadata["site"] = _canonical_site(bids.group(3))
        return metadata
    strambo = re.match(r"(STRAMBO_\d+)_([A-Za-z]{2})_Y?(\d+)", stem)
    if strambo:
        metadata["subject_id"] = strambo.group(1)
        metadata["site"] = _canonical_site(strambo.group(2))
        metadata["session_id"] = normalize_session_id(f"Y{strambo.group(3)}" if "_Y" in stem else strambo.group(3))
    return metadata


def _role_from_filename(path: Path) -> str | None:
    stem = _volume_stem(path).lower()
    if re.search(r"(?:^|[_-])mask[_-]?seg(?:$|[_-])", stem) or re.search(r"(?:^|[_-])seg(?:$|[_-])", stem):
        return "bone_segmentation"
    if re.search(r"(?:^|[_-])mask[_-]?full(?:$|[_-])", stem) or "full_mask" in stem or "periosteal" in stem:
        return "periosteal_mask"
    if re.search(r"(?:^|[_-])mask[_-]?trab(?:$|[_-])", stem) or "trabecular" in stem:
        return "trabecular_mask"
    if re.search(r"(?:^|[_-])mask[_-]?cort(?:$|[_-])", stem) or "cortical" in stem:
        return "cortical_mask"
    if re.search(r"(?:^|[_-])(image|scan|source)(?:$|[_-])", stem):
        return "transformed_image"
    if (
        not re.search(r"(?:^|[_-])(mask|seg|full|trab|cort|map|label)(?:$|[_-])", stem)
        and _volume_name(path).lower().endswith((".aim", ".isq"))
    ):
        return "transformed_image"
    return None


def _canonical_site(site: str) -> str:
    normalized = str(site or "").strip().lower()
    shared = normalize_site(normalized)
    if shared:
        return shared
    aliases = {
        "rl": "radiusleft",
        "rr": "radiusright",
        "tl": "tibialeft",
        "tr": "tibiaright",
    }
    return aliases.get(normalized, normalized)


def _filter_site(site: str) -> str:
    normalized = str(site or "").strip().lower()
    return normalize_site(normalized) or normalized


def _sites_match(case_site: str, requested_site: str) -> bool:
    case_site = _filter_site(case_site)
    requested_site = _filter_site(requested_site)
    if not requested_site:
        return True
    if case_site == requested_site:
        return True
    return _site_family(case_site) == requested_site or _site_family(requested_site) == case_site


def _site_family(site: str) -> str:
    normalized = str(site or "").strip().lower()
    if normalized.startswith("radius"):
        return "radius"
    if normalized.startswith("tibia"):
        return "tibia"
    return normalized


def _is_generic_site(site: str) -> bool:
    return str(site or "").strip().lower() in {"", "radius", "tibia", "knee"}


def _session_key(session_id) -> str:
    value = str(session_id or "").strip()
    upper = value.upper()
    if upper.startswith("SES-"):
        upper = upper[4:]
    if upper.startswith("Y") and upper[1:].isdigit():
        upper = upper[1:]
    return upper.lstrip("0") or "0"


def _volume_stem(path: Path) -> str:
    name = _volume_name(path)
    lower = name.lower()
    for suffix in (".nii.gz", ".npy", ".aim", ".isq", ".nii"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _is_supported_volume(path: Path) -> bool:
    return _volume_name(path).lower().endswith((".npy", ".nii", ".nii.gz", ".aim", ".isq"))


def _volume_name(path: Path) -> str:
    return path.name.split(";", 1)[0]


def _matching_record(records, role: str, key):
    exact = next((record for record in records if record.role == role and _case_key(record) == key), None)
    if exact is not None:
        return exact
    if role != "scan_region_native_common":
        return None
    return next(
        (
            record
            for record in records
            if record.role == role and _compatible_case_key(_case_key(record), key)
        ),
        None,
    )


def _matching_image_record(records, key):
    return next(
        (
            record
            for record in records
            if record.role in {"transformed_image", "source_image_view"} and _case_key(record) == key
        ),
        None,
    )


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
    return record.subject_id, _filter_site(record.site), record.session_id, record.stack_index, record.space


def _compatible_case_key(
    candidate: tuple[str, str, str | None, int | None, str],
    requested: tuple[str, str, str | None, int | None, str],
) -> bool:
    if candidate[:3] != requested[:3] or candidate[4] != requested[4]:
        return False
    return _compatible_stack_index(candidate[3], requested[3])


def _compatible_stack_index(left: int | None, right: int | None) -> bool:
    return left == right


def _load_volume(path: Path, *, scaling: str = "bmd") -> _LoadedVolume:
    if path.suffix == ".npy":
        return _LoadedVolume(np.load(path, allow_pickle=False))
    if path.name.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        return _LoadedVolume(
            sitk.GetArrayFromImage(image),
            tuple(reversed(tuple(float(value) for value in image.GetSpacing()))),
            image,
        )
    if _volume_name(path).lower().endswith(".aim"):
        return _load_aim_volume(path, scaling=scaling)
    raise ValueError(f"Unsupported image format: {path}")


def _load_record_volume(record: DerivativeRecord) -> _LoadedVolume:
    if record.source == "virtual" and record.role == "source_image_view":
        return _load_virtual_image_record(record)
    scaling = "bmd" if record.content_type == "image" else "native"
    return _load_volume(record.path, scaling=scaling)


def _load_aim_volume(path: Path, *, scaling: str = "bmd") -> _LoadedVolume:
    try:
        import py_aimio
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("AIM batch inputs require aimio-py and SimpleITK.") from exc

    array, metadata = py_aimio.read_aim(str(path), density=False, hu=False)
    array = _aim_array_zyx(np.asarray(array), metadata)
    processing_log = str(metadata.get("processing_log_raw") or metadata.get("processing_log", ""))
    array = _scale_aim_array(array, processing_log, scaling)
    image = sitk.GetImageFromArray(array)
    spacing_xyz = _aim_spacing_xyz(metadata)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(_aim_origin_xyz(metadata, spacing_xyz))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return _LoadedVolume(
        sitk.GetArrayFromImage(image),
        tuple(reversed(tuple(float(value) for value in image.GetSpacing()))),
        image,
    )


def _load_virtual_image_record(record: DerivativeRecord) -> _LoadedVolume:
    metadata = dict(record.metadata)
    source_image = Path(str(metadata.get("source_image") or record.path))
    if str(source_image).lower().endswith(".aim"):
        return _load_virtual_aim_stack(source_image, metadata)
    raise ValueError(f"Unsupported virtual image source: {source_image}")


def _load_virtual_aim_stack(source_image: Path, metadata: dict) -> _LoadedVolume:
    try:
        import py_aimio
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("Virtual AIM image records require aimio-py and SimpleITK.") from exc

    scaling = str(metadata.get("scaling") or "bmd")
    array, aim_metadata = py_aimio.read_aim(str(source_image), density=False, hu=False)
    array = _aim_array_zyx(np.asarray(array), aim_metadata)
    array = _scale_aim_array(array, str(aim_metadata.get("processing_log_raw") or aim_metadata.get("processing_log", "")), scaling)
    image = sitk.GetImageFromArray(array)
    spacing_xyz = _aim_spacing_xyz(aim_metadata)
    image.SetSpacing(spacing_xyz)
    origin_xyz = _aim_origin_xyz(aim_metadata, spacing_xyz)
    image.SetOrigin(origin_xyz)
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    crop = metadata.get("crop") if isinstance(metadata.get("crop"), dict) else {}
    if crop.get("applied"):
        index = crop.get("applied_roi_index_xyz")
        size = crop.get("applied_roi_size_xyz")
        if not (isinstance(index, list) and isinstance(size, list) and len(index) == 3 and len(size) == 3):
            raise ValueError("Virtual AIM crop metadata must include applied_roi_index_xyz and applied_roi_size_xyz")
        image = _sitk_crop_with_padding(image, tuple(int(v) for v in index), tuple(int(v) for v in size))
        image.SetOrigin((0.0, 0.0, 0.0))

    source_stack_index = metadata.get("source_stack_index")
    stack_depth = metadata.get("import_stack_depth")
    if source_stack_index is not None and stack_depth is not None:
        origin = list(image.GetOrigin())
        origin[2] += float(max(0, int(source_stack_index) - 1) * int(stack_depth)) * float(image.GetSpacing()[2])
        image.SetOrigin(tuple(origin))

    start = int(metadata["slice_start"])
    stop = int(metadata["slice_stop"])
    size = list(image.GetSize())
    size[2] = stop - start
    image = sitk.RegionOfInterest(image, size=size, index=[0, 0, start])
    return _LoadedVolume(
        sitk.GetArrayFromImage(image),
        tuple(reversed(tuple(float(value) for value in image.GetSpacing()))),
        image,
    )


def _aim_array_zyx(array: np.ndarray, metadata: dict) -> np.ndarray:
    dimensions = metadata.get("dimensions")
    if isinstance(dimensions, (list, tuple)) and len(dimensions) == 3:
        xyz = tuple(int(value) for value in dimensions)
        zyx = (xyz[2], xyz[1], xyz[0])
        if tuple(array.shape) == zyx:
            return array
        if tuple(array.shape) == xyz:
            return np.transpose(array, (2, 1, 0))
    return array


def _aim_spacing_xyz(metadata: dict) -> tuple[float, float, float]:
    spacing = metadata.get("spacing")
    if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
        return tuple(float(value) for value in spacing)
    element_size = metadata.get("element_size")
    if isinstance(element_size, (list, tuple)) and len(element_size) == 3:
        return tuple(float(value) for value in element_size)
    return (1.0, 1.0, 1.0)


def _aim_origin_xyz(metadata: dict, spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    origin = metadata.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) == 3:
        return tuple(float(value) for value in origin)
    position = metadata.get("position")
    if isinstance(position, (list, tuple)) and len(position) == 3:
        offset = metadata.get("offset", (0, 0, 0))
        if not (isinstance(offset, (list, tuple)) and len(offset) == 3):
            offset = (0, 0, 0)
        return tuple((float(position[i]) + float(offset[i]) + 0.5) * spacing_xyz[i] for i in range(3))
    return (0.0, 0.0, 0.0)


def _scale_aim_array(array: np.ndarray, processing_log: str, scaling: str) -> np.ndarray:
    scaling = scaling.lower()
    if scaling in {"native", "none"}:
        return array
    mu_scaling, hu_mu_water, hu_mu_air, density_slope, density_intercept = _aim_calibration(processing_log)
    values = array.astype(np.float32, copy=False)
    if scaling == "mu":
        return values / float(mu_scaling)
    if scaling == "hu":
        return values * (1000.0 / (mu_scaling * (hu_mu_water - hu_mu_air))) - 1000.0 * hu_mu_water / (hu_mu_water - hu_mu_air)
    if scaling in {"bmd", "density"}:
        return values / float(mu_scaling) * float(density_slope) + float(density_intercept)
    raise ValueError(f"Unsupported AIM scaling: {scaling}")


def _aim_calibration(processing_log: str) -> tuple[int, float, float, float, float]:
    patterns = {
        "mu": r"Mu_Scaling\s+(\d+)",
        "water": r"HU: mu water\s+(\d+\.\d+)",
        "slope": r"Density: slope\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
        "intercept": r"Density: intercept\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",
    }
    matches = {key: re.search(pattern, processing_log) for key, pattern in patterns.items()}
    if not all(matches.values()):
        raise ValueError("Could not parse AIM calibration constants from processing log.")
    return (
        int(matches["mu"].group(1)),
        float(matches["water"].group(1)),
        0.0,
        float(matches["slope"].group(1)),
        float(matches["intercept"].group(1)),
    )


def _sitk_crop_with_padding(image, index_xyz: tuple[int, int, int], size_xyz: tuple[int, int, int]):
    import SimpleITK as sitk

    image_size = image.GetSize()
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]
    for i in range(3):
        start = int(index_xyz[i])
        end = int(index_xyz[i] + size_xyz[i])
        if start < 0:
            pad_lower[i] = -start
        if end > image_size[i]:
            pad_upper[i] = end - image_size[i]
    if any(pad_lower) or any(pad_upper):
        image = sitk.ConstantPad(image, padLowerBound=pad_lower, padUpperBound=pad_upper, constant=0.0)
        index_xyz = tuple(int(index_xyz[i] + pad_lower[i]) for i in range(3))
    return sitk.RegionOfInterest(image, size=[int(v) for v in size_xyz], index=[int(v) for v in index_xyz])


def _write_map(path: Path, array: np.ndarray, reference: _LoadedVolume) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reference.sitk_image is None:
        np.save(path, np.asarray(array, dtype=np.float32))
        return
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.CopyInformation(reference.sitk_image)
    sitk.WriteImage(image, str(path))


def _load_map(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.name.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    raise ValueError(f"Unsupported map format: {path}")


def _read_existing_records(root: Path) -> tuple[DerivativeRecord, ...]:
    path = manifest_path(root, "Microarchitecture")
    return read_manifest(path).records if path.exists() else ()


def _output_case_key(record: DerivativeRecord) -> tuple[str, str, str | None, int | None]:
    return record.subject_id, record.site, record.session_id, record.stack_index


def _compatibility_hash(input_ids, spacing, use_common_region, thickness_method, thickness_backend) -> str:
    payload = {
        "inputs": list(input_ids),
        "spacing": [float(value) for value in spacing],
        "use_common_region": bool(use_common_region),
        "thickness_method": str(thickness_method),
        "thickness_backend": str(thickness_backend),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _expected_output_signatures(case: dict[str, DerivativeRecord]) -> set[tuple[str, str, str | None]]:
    names = {"Tb.Th", "Tb.Sp", "Tb.N", "Tt.BMD", "Tb.BMD"}
    if "cortical_mask" in case:
        names.update({"Ct.Th", "Ct.Po.Dm", "Ct.BMD"})
    return {("measurements_table", "table", None)} | {
        (_MAP_ROLES.get(name, "material_map"), "native", name) for name in names
    }


def _expected_native_map_signatures(case: dict[str, DerivativeRecord]) -> set[tuple[str, str, str | None]]:
    return {signature for signature in _expected_output_signatures(case) if signature[0] != "measurements_table"}


def _record_signature(record: DerivativeRecord) -> tuple[str, str, str | None]:
    return record.role, record.space, record.metadata.get("map_name")


def _record_replacement_signature(record: DerivativeRecord) -> tuple[str, str, str | None, bool]:
    return (*_record_signature(record), bool(record.metadata.get("use_common_region")))


def _find_compatible_measurement_record(existing, case_key, input_ids, settings_hash, case):
    required = {("measurements_table", "table", None)}
    matching = [
        record for record in existing
        if _output_case_key(record) == case_key
        and record.inputs == input_ids
        and record.settings_hash == settings_hash
        and record.path.is_file()
    ]
    by_signature = {_record_signature(record): record for record in matching}
    if not required <= set(by_signature):
        return None
    measurement_record = by_signature[("measurements_table", "table", None)]
    if not _measurement_csv_has_required_parameters(measurement_record.path, case):
        return None
    return measurement_record


def _find_compatible_native_map_records(existing, case_key, input_ids, settings_hash, case):
    required = _expected_native_map_signatures(case)
    matching = [
        record for record in existing
        if _output_case_key(record) == case_key
        and record.inputs == input_ids
        and record.settings_hash == settings_hash
        and record.path.is_file()
    ]
    by_signature = {_record_signature(record): record for record in matching}
    if not required <= set(by_signature):
        return None
    return {record.metadata["map_name"]: record for record in by_signature.values() if record.metadata.get("map_name")}


def _load_native_maps_from_records(records: dict[str, DerivativeRecord]) -> dict[str, np.ndarray] | None:
    try:
        return {name: _load_map(record.path) for name, record in records.items()}
    except Exception:
        return None


def _summarize_measurements(grayscale, masks, maps, spacing) -> dict[str, float]:
    bone = np.asarray(masks["bone_segmentation"]) > 0
    peri = np.asarray(masks["periosteal_mask"]) > 0
    trab = np.asarray(masks["trabecular_mask"]) > 0
    cort = np.asarray(masks["cortical_mask"]) > 0 if "cortical_mask" in masks else None
    trab_region = trab & peri
    if cort is not None:
        trab_region = trab_region & ~cort
        cort_region = cort & peri
    else:
        cort_region = None
    trab_bone = bone & trab_region
    tb_th = summary(np.asarray(maps["Tb.Th"])[trab_bone])
    metrics = compartment_metrics(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab,
        cortical_mask=cort,
        spacing=spacing,
        mean_tb_th=tb_th["mean"],
    )
    tb_sp_values = np.asarray(maps["Tb.Sp"])[trab_region & ~trab_bone]
    tb_sp = summary(tb_sp_values)
    tb_n_map = np.asarray(maps["Tb.N"])
    tb_n = summary(tb_n_map[trab_region])
    tb_inverse_number_map = np.zeros(trab_region.shape, dtype=np.float32)
    valid_tb_n = trab_region & np.isfinite(tb_n_map) & (tb_n_map > 0)
    tb_inverse_number_map[valid_tb_n] = (1.0 / tb_n_map[valid_tb_n]).astype(np.float32, copy=False)
    tb_inverse_number = summary(tb_inverse_number_map[valid_tb_n])
    metrics.update(
        {
            "Tb.Th": tb_th["mean"],
            "Tb.Th SD": tb_th["sd"],
            "Tb.Th Min": tb_th["min"],
            "Tb.Th Max": tb_th["max"],
            "Tb.Sp": tb_sp["mean"],
            "Tb.Sp SD": tb_sp["sd"],
            "Tb.Sp Min": tb_sp["min"],
            "Tb.Sp Max": tb_sp["max"],
            "Tb.N": tb_n["mean"],
            "Tb.N Median": tb_n["median"],
            "Tb.N SD": tb_n["sd"],
            "Tb.N P5": tb_n["p5"],
            "Tb.N P25": tb_n["p25"],
            "Tb.N P75": tb_n["p75"],
            "Tb.N P95": tb_n["p95"],
            "Tb.N Min": tb_n["min"],
            "Tb.N Max": tb_n["max"],
            "Tb.1/N.SD": tb_inverse_number["sd"],
        }
    )
    image = np.asarray(grayscale, dtype=np.float32)
    tt_mean, tt_sd = masked_mean_sd(image, peri)
    tb_mean, tb_sd = masked_mean_sd(image, trab_region)
    metrics.update({"Tt.BMD": tt_mean, "Tt.BMD SD": tt_sd, "Tb.BMD": tb_mean, "Tb.BMD SD": tb_sd})
    if cort_region is not None:
        cort_bone = bone & cort_region
        ct_th = summary(np.asarray(maps["Ct.Th"])[cort_bone])
        pore_summary = summary(np.asarray(maps["Ct.Po.Dm"])[cort_region & ~cort_bone])
        ct_mean, ct_sd = masked_mean_sd(image, cort_region)
        metrics.update(
            {
                "Ct.Th": ct_th["mean"],
                "Ct.Th SD": ct_th["sd"],
                "Ct.Th Min": ct_th["min"],
                "Ct.Th Max": ct_th["max"],
                "Ct.Po.Dm": pore_summary["mean"],
                "Ct.Po.Dm SD": pore_summary["sd"],
                "Ct.Po.Dm Min": pore_summary["min"],
                "Ct.Po.Dm Max": pore_summary["max"],
                "Ct.BMD": ct_mean,
                "Ct.BMD SD": ct_sd,
            }
        )
    return metrics


def _masked_maps_for_measurements(maps, masks) -> dict[str, np.ndarray]:
    peri = np.asarray(masks["periosteal_mask"]) > 0
    trab = np.asarray(masks["trabecular_mask"]) > 0
    cort = np.asarray(masks["cortical_mask"]) > 0 if "cortical_mask" in masks else None
    trab_region = trab & peri & ~(cort if cort is not None else np.zeros_like(trab, dtype=bool))
    result = {
        "Tb.Th": np.where(trab_region, maps["Tb.Th"], 0),
        "Tb.Sp": np.where(trab_region, maps["Tb.Sp"], 0),
        "Tb.N": np.where(trab_region, maps["Tb.N"], 0),
    }
    if "Tt.BMD" in maps:
        result["Tt.BMD"] = np.where(peri, maps["Tt.BMD"], 0)
    if "Tb.BMD" in maps:
        result["Tb.BMD"] = np.where(trab_region, maps["Tb.BMD"], 0)
    if cort is not None:
        cort_region = cort & peri
        for name in ("Ct.Th", "Ct.Po.Dm", "Ct.BMD"):
            if name in maps:
                result[name] = np.where(cort_region, maps[name], 0)
    return result


def _measurement_csv_has_required_parameters(path: Path, case: dict[str, DerivativeRecord]) -> bool:
    required = {"Tt.BMD", "Tb.BMD", "Tb.BV/TV", "Tb.Th", "Tb.Sp", "Tb.N"}
    if "cortical_mask" in case:
        required.update({"Ct.BMD", "Ct.Th", "Ct.Po", "Ct.Po.Dm"})
    try:
        import csv

        with Path(path).open(newline="", encoding="utf-8") as handle:
            present = {row.get("Parameter", "") for row in csv.DictReader(handle)}
    except Exception:
        return False
    return required <= present


def _emit(progress, subject_id, site, session_id, step, status, message, path=None) -> None:
    event = DerivativeProgressEvent("Microarchitecture", subject_id, site, session_id, step, status, message, path)
    if progress is None:
        print(format_progress_event(event))
    else:
        progress(event)

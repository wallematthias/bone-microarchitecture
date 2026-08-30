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
    discover_manifests,
    format_progress_event,
    read_manifest,
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
_MAP_ROLES = {
    "Tb.Th": "trabecular_thickness_map",
    "Tb.Sp": "trabecular_spacing_map",
    "Tb.N": "trabecular_number_map",
    "Ct.Th": "cortical_thickness_map",
    "Ct.Po.Dm": "cortical_porosity_map",
    "Tb.BMD": "material_map",
    "Ct.BMD": "material_map",
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
        settings_hash = _compatibility_hash(input_ids, case_spacing, use_common_region, thickness_method, thickness_backend)
        case_key = _output_case_key(case["bone_segmentation"])
        reused = None if force else _find_compatible_measurement_record(
            existing_records, case_key, input_ids, settings_hash, case
        )
        if reused is not None:
            measurement_records.append(reused)
            _emit(progress, subject_id, site, session_id, "measure", "reused", "Reused compatible measurements", reused.path)
            continue

        _emit(progress, subject_id, site, session_id, "measure", "started", "Computing microarchitecture")
        masks = {role: _load_record_volume(record).array for role, record in case.items() if role != "transformed_image"}
        common = masks.pop("scan_region_native_common", None) if use_common_region else None
        if common is not None:
            common = np.asarray(common) > 0
            masks = {role: (np.asarray(mask) > 0) & common for role, mask in masks.items()}

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
        session_part = f"ses-{session_id}" if session_id else "ses-unknown"
        filename = f"sub-{subject_id}_{session_part}_site-{site}_measurements.csv"
        output_path = record_output_path(
            root, "Microarchitecture", subject_id, site, "native_space", session_part, "measurements", filename
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_measurement_csv(output_path, result.measurements, result.maps)
        record_metadata = {
            "use_common_region": common is not None,
            "thickness_method": thickness_method,
            "thickness_backend": result.metadata["thickness_backend"],
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
        for map_name, map_array in result.maps.items():
            role = _MAP_ROLES.get(map_name, "material_map")
            extension = ".nii.gz" if image.sitk_image is not None else ".npy"
            map_filename = f"sub-{subject_id}_{session_part}_site-{site}_map-{map_name.lower().replace('.', '-')}{extension}"
            map_path = record_output_path(
                root, "Microarchitecture", subject_id, site, "native_space", session_part, "maps", map_filename
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
                    inputs=input_ids,
                    metadata={**record_metadata, "map_name": map_name},
                    content_type="image",
                    settings_hash=settings_hash,
                )
            )
        _emit(progress, subject_id, site, session_id, "measure", "completed", "Wrote measurements", output_path)

    if output_records:
        superseded = {(_output_case_key(record), _record_signature(record)) for record in output_records}
        merged_records = [
            record for record in existing_records
            if (_output_case_key(record), _record_signature(record)) not in superseded
        ]
        merged_records.extend(output_records)
        manifest = DerivativeManifest.create(
            "Microarchitecture", root, {"name": "bone-microarchitecture", "version": "0.2.2"},
            records=tuple(merged_records),
        )
        write_manifest(manifest, manifest_path(root, "Microarchitecture"))
    return measurement_records


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


def _load_volume(path: Path) -> _LoadedVolume:
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
    raise ValueError(f"Unsupported image format: {path}")


def _load_record_volume(record: DerivativeRecord) -> _LoadedVolume:
    if record.source == "virtual" and record.role == "source_image_view":
        return _load_virtual_image_record(record)
    return _load_volume(record.path)


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
    names = {"Tb.Th", "Tb.Sp", "Tb.N", "Tb.BMD"}
    if "cortical_mask" in case:
        names.update({"Ct.Th", "Ct.Po.Dm", "Ct.BMD"})
    return {("measurements_table", "table", None)} | {
        (_MAP_ROLES.get(name, "material_map"), "native", name) for name in names
    }


def _record_signature(record: DerivativeRecord) -> tuple[str, str, str | None]:
    return record.role, record.space, record.metadata.get("map_name")


def _find_compatible_measurement_record(existing, case_key, input_ids, settings_hash, case):
    required = _expected_output_signatures(case)
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
    return by_signature[("measurements_table", "table", None)]


def _emit(progress, subject_id, site, session_id, step, status, message, path=None) -> None:
    event = DerivativeProgressEvent("Microarchitecture", subject_id, site, session_id, step, status, message, path)
    if progress is None:
        print(format_progress_event(event))
    else:
        progress(event)

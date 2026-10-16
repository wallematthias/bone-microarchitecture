"""File-level functional-bone analysis with region-specific final-map caching."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .batch import _load_volume, _mask_on_reference_grid, _write_map
from .cache import cache_is_current, input_fingerprints, output_file_states
from .method import METHOD_METADATA
from .pipeline import compute_microarchitecture
from .results import write_measurement_csv


def run_microarchitecture_case(
    *,
    image_path,
    bone_mask_path,
    periosteal_mask_path,
    trabecular_mask_path,
    cortical_mask_path,
    output_dir,
    prefix,
    common_region_path=None,
    voidspace_mask_path=None,
    thickness_method="hildebrand",
    thickness_backend="auto",
    force=False,
):
    """Calculate a file-aligned region once, saving its maps beside its table.

    The optional functional ROI is periosteal ∩ common ∩ not-voidspace. Full
    native arrays remain the distance/pore context. Only complete compatible
    final outputs are reused; no distance fields or native maps are cached here.
    """
    paths = dict(
        transformed_image=image_path,
        bone_segmentation=bone_mask_path,
        periosteal_mask=periosteal_mask_path,
        trabecular_mask=trabecular_mask_path,
        cortical_mask=cortical_mask_path,
    )
    if common_region_path is not None:
        paths["scan_region_native_common"] = common_region_path
    if voidspace_mask_path is not None:
        paths["voidspace_mask"] = voidspace_mask_path
    fingerprints = input_fingerprints(paths.values())
    input_paths = {role: str(Path(path).resolve()) for role, path in paths.items()}
    settings = {
        **METHOD_METADATA,
        "input_fingerprints": fingerprints,
        "input_paths": input_paths,
        "thickness_method": thickness_method,
        "thickness_backend_request": thickness_backend,
        "has_common": common_region_path is not None,
        "has_voidspace": voidspace_mask_path is not None,
        "mask_grid_policy": "physical-nearest-neighbor-v1",
    }
    settings_hash = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    output_dir = Path(output_dir)
    output = output_dir / f"{prefix}_measurements.csv"
    if not force and output.is_file():
        try:
            metadata = json.loads(output.with_suffix(".json").read_text())
            if (
                metadata.get("settings_hash") == settings_hash
                and str(output.resolve()) in metadata.get("output_file_states", {})
                and cache_is_current(metadata)
            ):
                return output
        except (OSError, ValueError, TypeError):
            pass

    image = _load_volume(Path(image_path), scaling="bmd")
    if image.spacing is None:
        raise ValueError("Region microarchitecture needs image geometry/spacing")
    masks = {
        role: _mask_on_reference_grid(
            _load_volume(Path(paths[role]), scaling="native"), image
        )
        for role in (
            "bone_segmentation",
            "periosteal_mask",
            "trabecular_mask",
            "cortical_mask",
        )
    }
    analysis = masks["periosteal_mask"].copy()
    if common_region_path is not None:
        analysis &= _mask_on_reference_grid(
            _load_volume(Path(common_region_path), scaling="native"), image
        )
    if voidspace_mask_path is not None:
        analysis &= ~_mask_on_reference_grid(
            _load_volume(Path(voidspace_mask_path), scaling="native"), image
        )
    result = compute_microarchitecture(
        grayscale=image.array,
        spacing=image.spacing,
        bone_mask=masks["bone_segmentation"],
        periosteal_mask=masks["periosteal_mask"],
        trabecular_mask=masks["trabecular_mask"],
        cortical_mask=masks["cortical_mask"],
        analysis_mask=analysis,
        thickness_method=thickness_method,
        thickness_backend=thickness_backend,
    )
    print(
        f"[microarchitecture] backend={result.metadata['thickness_backend']}; "
        f"device={result.metadata.get('thickness_device')}; GPU={result.metadata.get('thickness_gpu')}",
        flush=True,
    )
    maps_dir = output_dir / "maps"
    extension = ".nii.gz" if image.sitk_image is not None else ".npy"
    files = []
    for name, array in result.maps.items():
        path = maps_dir / f"{prefix}_map-{name.lower().replace('.', '-')}{extension}"
        _write_map(path, array, image)
        files.append(path)
    label = "functional" if voidspace_mask_path is not None else "native"
    if common_region_path is not None:
        label = "functional-common" if voidspace_mask_path is not None else "common"
    if voidspace_mask_path is not None:
        path = output_dir / f"{prefix}_desc-functional-bone-analysis_mask{extension}"
        _write_analysis_mask(path, analysis, image)
        files.append(path)
    metadata = {
        **result.metadata,
        "settings_hash": settings_hash,
        "input_paths": input_paths,
        "input_fingerprints": fingerprints,
        "analysis_region": label,
        "use_common_region": common_region_path is not None,
        "functional_bone": voidspace_mask_path is not None,
    }
    # Publish cache metadata last: an interrupted calculation cannot certify a
    # partial set of maps, or accidentally reuse an older result after a rerun.
    write_measurement_csv(output, result.measurements, result.maps, metadata=metadata)
    metadata["output_file_states"] = output_file_states([*files, output])
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def _write_analysis_mask(path, array, reference):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reference.sitk_image is None:
        np.save(path, np.asarray(array, dtype=np.uint8))
    else:
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
        image.CopyInformation(reference.sitk_image)
        sitk.WriteImage(image, str(path))

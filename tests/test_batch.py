from __future__ import annotations

import csv

import numpy as np
import pytest

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest


def _record(root, role, path, *, derivative="Segmentation"):
    return DerivativeRecord(
        derivative=derivative,
        role=role,
        subject_id="SAMPLE001",
        site="tibia",
        session_id="1",
        stack_index=None,
        space="native",
        path=root / path,
        source="provided",
        content_type="image" if role == "transformed_image" else "mask",
    )


def test_batch_discovers_manifest_inputs_clips_common_region_and_writes_measurement_manifest(tmp_path):
    """A missing common-region intersection would inflate the exported Tb.TV."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    biological_mask = np.ones((4, 4, 4), dtype=np.uint8)
    common_region = np.zeros((4, 4, 4), dtype=np.uint8)
    common_region[1:3, 1:3, 1:3] = 1

    paths = {
        "image": "inputs/sub-SAMPLE001_ses-1_image.npy",
        "bone": "inputs/sub-SAMPLE001_ses-1_bone.npy",
        "peri": "inputs/sub-SAMPLE001_ses-1_peri.npy",
        "trab": "inputs/sub-SAMPLE001_ses-1_trab.npy",
        "common": "inputs/sub-SAMPLE001_ses-1_common.npy",
    }
    for key, array in {
        "image": image,
        "bone": biological_mask,
        "peri": biological_mask,
        "trab": biological_mask,
        "common": common_region,
    }.items():
        path = tmp_path / paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    segmentation = DerivativeManifest.create(
        "Segmentation",
        tmp_path,
        {"name": "test", "version": "1"},
        records=(
            _record(tmp_path, "transformed_image", paths["image"], derivative="Registration"),
            _record(tmp_path, "bone_segmentation", paths["bone"]),
            _record(tmp_path, "periosteal_mask", paths["peri"]),
            _record(tmp_path, "trabecular_mask", paths["trab"]),
        ),
    )
    common = DerivativeManifest.create(
        "CommonRegion",
        tmp_path,
        {"name": "test", "version": "1"},
        records=(_record(tmp_path, "scan_region_native_common", paths["common"], derivative="CommonRegion"),),
    )
    write_manifest(segmentation, tmp_path / "derivatives/Segmentation/manifest.json")
    write_manifest(common, tmp_path / "derivatives/CommonRegion/manifest.json")

    records = run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        thickness_method="edt",
        thickness_backend="cpu",
    )

    assert len(records) == 1
    manifest_path = tmp_path / "derivatives/Microarchitecture/manifest.json"
    assert manifest_path.is_file()
    csv_path = records[0].path
    assert csv_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["Parameter"]: row for row in csv.DictReader(handle)}
    assert float(rows["Tb.TV"]["Mean"]) == 8.0
    assert float(rows["Tb.BMD"]["Mean"]) == 100.0
    assert records[0].role == "measurements_table"
    assert "/measurements/" in str(csv_path)
    manifest = read_manifest(manifest_path)
    map_records = [record for record in manifest.records if record.role != "measurements_table"]
    assert {record.role for record in map_records} == {
        "trabecular_thickness_map",
        "trabecular_spacing_map",
        "trabecular_number_map",
    }
    assert all(record.path.is_file() and "/maps/" in str(record.path) for record in map_records)


def test_batch_uses_nifti_geometry_and_common_region_for_measurement(tmp_path):
    """NIfTI geometry, not a unit-spacing default, determines physical volume."""
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    biological = np.ones((4, 4, 4), dtype=np.uint8)
    common = np.zeros((4, 4, 4), dtype=np.uint8)
    common[1:3, 1:3, 1:3] = 1
    paths = {
        "image": "inputs/image.nii.gz",
        "bone": "inputs/bone.nii.gz",
        "peri": "inputs/peri.nii.gz",
        "trab": "inputs/trab.nii.gz",
        "common": "inputs/common.nii.gz",
    }
    for key, array in {"image": image, "bone": biological, "peri": biological, "trab": biological, "common": common}.items():
        output = tmp_path / paths[key]
        output.parent.mkdir(parents=True, exist_ok=True)
        itk_image = sitk.GetImageFromArray(array)
        itk_image.SetSpacing((0.5, 1.0, 3.0))
        sitk.WriteImage(itk_image, str(output))

    inputs = DerivativeManifest.create(
        "Segmentation", tmp_path, {"name": "test", "version": "1"},
        records=(
            _record(tmp_path, "transformed_image", paths["image"], derivative="Registration"),
            _record(tmp_path, "bone_segmentation", paths["bone"]),
            _record(tmp_path, "periosteal_mask", paths["peri"]),
            _record(tmp_path, "trabecular_mask", paths["trab"]),
        ),
    )
    common_manifest = DerivativeManifest.create(
        "CommonRegion", tmp_path, {"name": "test", "version": "1"},
        records=(_record(tmp_path, "scan_region_native_common", paths["common"], derivative="CommonRegion"),),
    )
    write_manifest(inputs, tmp_path / "derivatives/Segmentation/manifest.json")
    write_manifest(common_manifest, tmp_path / "derivatives/CommonRegion/manifest.json")

    records = run_microarchitecture_batch(tmp_path, thickness_method="edt", thickness_backend="cpu")

    with records[0].path.open(newline="", encoding="utf-8") as handle:
        rows = {row["Parameter"]: row for row in csv.DictReader(handle)}
    assert float(rows["Tb.TV"]["Mean"]) == 12.0
    map_records = read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records[1:]
    assert all(record.path.suffixes[-2:] == [".nii", ".gz"] for record in map_records)


def test_batch_preserves_unrelated_microarchitecture_manifest_records(tmp_path):
    """A rerun for one case must not erase already-complete unrelated cases."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.ones((3, 3, 3), dtype=np.float32)
    for name in ("image", "bone_mask", "periosteal_mask", "trabecular_mask"):
        np.save(tmp_path / f"{name}.npy", image)
    preserved = DerivativeRecord(
        derivative="Microarchitecture",
        role="measurements_table",
        subject_id="SAMPLE002",
        site="radius",
        session_id="2",
        stack_index=None,
        space="table",
        path=tmp_path / "derivatives/Microarchitecture/sub-SAMPLE002/site-radius/native_space/ses-2/measurements/measurements.csv",
        source="generated",
        content_type="table",
    )
    write_manifest(
        DerivativeManifest.create("Microarchitecture", tmp_path, {"name": "test", "version": "1"}, records=(preserved,)),
        tmp_path / "derivatives/Microarchitecture/manifest.json",
    )

    run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")

    records = read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
    assert any(record.subject_id == "SAMPLE002" and record.role == "measurements_table" for record in records)


def test_npy_batch_requires_explicit_spacing(tmp_path):
    """Removing an explicit spacing for geometry-free arrays must stop the run."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.ones((3, 3, 3), dtype=np.float32)
    for name in ("image", "bone_mask", "periosteal_mask", "trabecular_mask"):
        np.save(tmp_path / f"{name}.npy", image)

    with pytest.raises(ValueError, match=r"spacing.*\.npy"):
        run_microarchitecture_batch(tmp_path, thickness_method="edt", thickness_backend="cpu")

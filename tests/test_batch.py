from __future__ import annotations

import csv
from dataclasses import replace

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
        "material_map",
        "trabecular_thickness_map",
        "trabecular_spacing_map",
        "trabecular_number_map",
    }
    assert all(record.path.is_file() and "/maps/" in str(record.path) for record in map_records)


def test_batch_excludes_unused_common_region_from_provenance(tmp_path):
    """Disabling common-region clipping must also omit it from output inputs."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.ones((4, 4, 4), dtype=np.float32)
    paths = {
        "image": "inputs/sub-SAMPLE001_ses-1_image.npy",
        "bone": "inputs/sub-SAMPLE001_ses-1_bone.npy",
        "peri": "inputs/sub-SAMPLE001_ses-1_peri.npy",
        "trab": "inputs/sub-SAMPLE001_ses-1_trab.npy",
        "common": "inputs/sub-SAMPLE001_ses-1_common.npy",
    }
    for path in paths.values():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, image)
    common_record = _record(tmp_path, "scan_region_native_common", paths["common"], derivative="CommonRegion")
    write_manifest(
        DerivativeManifest.create(
            "Segmentation",
            tmp_path,
            {"name": "test", "version": "1"},
            records=(
                _record(tmp_path, "transformed_image", paths["image"], derivative="Registration"),
                _record(tmp_path, "bone_segmentation", paths["bone"]),
                _record(tmp_path, "periosteal_mask", paths["peri"]),
                _record(tmp_path, "trabecular_mask", paths["trab"]),
            ),
        ),
        tmp_path / "derivatives/Segmentation/manifest.json",
    )
    write_manifest(
        DerivativeManifest.create("CommonRegion", tmp_path, {"name": "test", "version": "1"}, records=(common_record,)),
        tmp_path / "derivatives/CommonRegion/manifest.json",
    )

    records = run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        use_common_region=False,
        thickness_method="edt",
        thickness_backend="cpu",
    )

    assert common_record.record_id not in records[0].inputs
    output_records = read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
    assert all(common_record.record_id not in record.inputs for record in output_records)


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


def test_cli_uses_nifti_geometry_without_spacing_and_preserves_map_geometry(tmp_path):
    """CLI must pass omitted spacing through to geometry-backed NIfTI inputs."""
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture.cli import main

    paths = {role: f"inputs/{role}.nii.gz" for role in ("image", "bone", "peri", "trab")}
    reference = None
    for role, path in paths.items():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.float32))
        image.SetSpacing((0.4, 0.7, 1.3))
        image.SetOrigin((3.0, 4.0, 5.0))
        image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
        sitk.WriteImage(image, str(output))
        if role == "image":
            reference = image
    manifest = DerivativeManifest.create(
        "Segmentation", tmp_path, {"name": "test", "version": "1"},
        records=(
            _record(tmp_path, "transformed_image", paths["image"], derivative="Registration"),
            _record(tmp_path, "bone_segmentation", paths["bone"]),
            _record(tmp_path, "periosteal_mask", paths["peri"]),
            _record(tmp_path, "trabecular_mask", paths["trab"]),
        ),
    )
    write_manifest(manifest, tmp_path / "derivatives/Segmentation/manifest.json")

    assert main(["run-batch", str(tmp_path), "--thickness-method", "edt", "--thickness-backend", "cpu"]) == 0

    map_record = next(record for record in read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records if record.role == "trabecular_thickness_map")
    written = sitk.ReadImage(str(map_record.path))
    np.testing.assert_allclose(written.GetSpacing(), reference.GetSpacing())
    np.testing.assert_allclose(written.GetOrigin(), reference.GetOrigin())
    np.testing.assert_allclose(written.GetDirection(), reference.GetDirection())


def test_cli_npy_without_spacing_reaches_clear_batch_validation(tmp_path):
    """CLI must expose the API's clear .npy spacing error instead of TypeError."""
    from bone_microarchitecture.cli import main

    image = np.ones((3, 3, 3), dtype=np.float32)
    for name in ("image", "bone_mask", "periosteal_mask", "trabecular_mask"):
        np.save(tmp_path / f"{name}.npy", image)

    with pytest.raises(ValueError, match=r"spacing.*\.npy"):
        main(["run-batch", str(tmp_path), "--thickness-method", "edt", "--thickness-backend", "cpu"])


def test_batch_reuses_compatible_records_and_recomputes_when_settings_or_inputs_change(tmp_path, monkeypatch):
    """Only matching input IDs and settings may reuse a complete output group."""
    import bone_microarchitecture.batch as batch

    image = np.ones((3, 3, 3), dtype=np.float32)
    paths = {role: f"inputs/{role}.npy" for role in ("image", "bone", "peri", "trab")}
    for path in paths.values():
        output = tmp_path / path
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, image)
    inputs = [
        _record(tmp_path, "transformed_image", paths["image"], derivative="Registration"),
        _record(tmp_path, "bone_segmentation", paths["bone"]),
        _record(tmp_path, "periosteal_mask", paths["peri"]),
        _record(tmp_path, "trabecular_mask", paths["trab"]),
    ]
    write_manifest(
        DerivativeManifest.create("Segmentation", tmp_path, {"name": "test", "version": "1"}, records=tuple(inputs)),
        tmp_path / "derivatives/Segmentation/manifest.json",
    )
    batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")
    output_manifest = tmp_path / "derivatives/Microarchitecture/manifest.json"
    before_mtime = output_manifest.stat().st_mtime_ns

    def should_not_compute(**kwargs):
        raise AssertionError("compatible outputs should be reused")

    monkeypatch.setattr(batch, "compute_microarchitecture", should_not_compute)
    batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")
    assert output_manifest.stat().st_mtime_ns == before_mtime

    original = __import__("bone_microarchitecture.pipeline", fromlist=["compute_microarchitecture"]).compute_microarchitecture
    calls = []

    def counted_compute(**kwargs):
        calls.append(kwargs["thickness_method"])
        return original(**kwargs)

    monkeypatch.setattr(batch, "compute_microarchitecture", counted_compute)
    batch.run_microarchitecture_batch(
        tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu", force=True
    )
    assert calls == ["edt"]

    calls.clear()
    batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="hildebrand", thickness_backend="cpu")
    assert calls == ["hildebrand"]

    preserved = DerivativeRecord(
        derivative="Microarchitecture", role="qc_report", subject_id="SAMPLE001", site="tibia", session_id="1",
        stack_index=None, space="native", path=tmp_path / "qc.txt", source="generated", content_type="report",
    )
    current = read_manifest(output_manifest)
    write_manifest(DerivativeManifest.create("Microarchitecture", tmp_path, current.software, records=(*current.records, preserved)), output_manifest)
    changed_inputs = list(inputs)
    changed_inputs[0] = replace(changed_inputs[0], record_id="different-image-record")
    write_manifest(DerivativeManifest.create("Segmentation", tmp_path, {"name": "test", "version": "1"}, records=tuple(changed_inputs)), tmp_path / "derivatives/Segmentation/manifest.json")
    calls.clear()
    batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="hildebrand", thickness_backend="cpu")
    records = read_manifest(output_manifest).records
    assert calls == ["hildebrand"]
    assert preserved in records


def test_grayscale_batch_persists_bmd_maps_with_deterministic_generic_role(tmp_path):
    """Every BMD map returned by the core pipeline must be discoverable in the manifest."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.full((4, 4, 4), 250.0, dtype=np.float32)
    for name in ("image", "bone_mask", "periosteal_mask", "trabecular_mask", "cortical_mask"):
        np.save(tmp_path / f"{name}.npy", image)

    run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")

    bmd_records = [
        record for record in read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
        if record.role == "material_map" and record.metadata.get("map_name") in {"Tb.BMD", "Ct.BMD"}
    ]
    assert {record.metadata["map_name"] for record in bmd_records} == {"Tb.BMD", "Ct.BMD"}
    assert all(record.path.is_file() for record in bmd_records)

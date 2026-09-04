from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, read_manifest, write_manifest


def _record(root, role, path, *, derivative="Segmentation", source="provided", content_type=None, metadata=None):
    return DerivativeRecord(
        derivative=derivative,
        role=role,
        subject_id="SAMPLE001",
        site="tibia",
        session_id="1",
        stack_index=None,
        space="native",
        path=root / path,
        source=source,
        content_type=content_type or ("image" if role in {"transformed_image", "source_image_view"} else "mask"),
        metadata=metadata or {},
    )


def _write_case(root, *, subject_id="SAMPLE001", site="tibia", session_id="1"):
    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    paths = {
        "transformed_image": f"inputs/sub-{subject_id}_ses-{session_id}_site-{site}_image.npy",
        "bone_segmentation": f"inputs/sub-{subject_id}_ses-{session_id}_site-{site}_bone.npy",
        "periosteal_mask": f"inputs/sub-{subject_id}_ses-{session_id}_site-{site}_peri.npy",
        "trabecular_mask": f"inputs/sub-{subject_id}_ses-{session_id}_site-{site}_trab.npy",
    }
    for role, path_text in paths.items():
        path = root / path_text
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, image if role == "transformed_image" else mask)
    manifest_path = root / "derivatives" / f"Segmentation-{subject_id}-{site}-{session_id}" / "manifest.json"
    write_manifest(
        DerivativeManifest.create(
            "Segmentation",
            root,
            {"name": "test", "version": "1"},
            records=tuple(
                replace(
                    _record(root, role, path_text, derivative="Registration" if role == "transformed_image" else "Segmentation"),
                    subject_id=subject_id,
                    site=site,
                    session_id=session_id,
                )
                for role, path_text in paths.items()
            ),
        ),
        manifest_path,
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
    assert "/xct/registered_measurements/" in str(csv_path)
    manifest = read_manifest(manifest_path)
    map_records = [record for record in manifest.records if record.role != "measurements_table"]
    assert {record.role for record in map_records} == {
        "material_map",
        "trabecular_thickness_map",
        "trabecular_spacing_map",
        "trabecular_number_map",
    }
    assert all(record.path.is_file() and "/xct/maps/" in str(record.path) for record in map_records)
    assert not any("/xct/registered/maps/" in str(record.path) for record in manifest.records)


def test_batch_loads_virtual_aim_source_image_view(monkeypatch, tmp_path):
    """Batch mode should not require Timelapsed to materialize split grayscale stack NIfTIs."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    for name in ("bone", "peri", "trab"):
        path = tmp_path / f"{name}.npy"
        np.save(path, mask)

    source = tmp_path / "raw" / "scan.AIM"
    calls = []

    def fake_load_virtual(record):
        calls.append(record.path)
        return batch._LoadedVolume(image, (0.061, 0.061, 0.061), None)

    monkeypatch.setattr(batch, "_load_virtual_image_record", fake_load_virtual)
    write_manifest(
        DerivativeManifest.create(
            "Segmentation",
            tmp_path,
            {"name": "test", "version": "1"},
            records=(
                _record(
                    tmp_path,
                    "source_image_view",
                    source,
                    source="virtual",
                    content_type="image",
                    metadata={
                        "format": "AIM",
                        "view_type": "stack_slices",
                        "slice_axis": "z",
                        "slice_start": 0,
                        "slice_stop": 4,
                        "source_image": str(source),
                        "scaling": "bmd",
                    },
                ),
                _record(tmp_path, "bone_segmentation", "bone.npy"),
                _record(tmp_path, "periosteal_mask", "peri.npy"),
                _record(tmp_path, "trabecular_mask", "trab.npy"),
            ),
        ),
        tmp_path / "derivatives/Segmentation/manifest.json",
    )

    records = batch.run_microarchitecture_batch(tmp_path, thickness_backend="cpu")

    assert records
    assert calls == [source]


def test_cli_module_execution_runs_batch(tmp_path):
    """The module entry point should behave like the console script."""
    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    for name, array in {
        "image": image,
        "bone": mask,
        "peri": mask,
        "trab": mask,
    }.items():
        path = tmp_path / "inputs" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    manifest = DerivativeManifest.create(
        "Segmentation",
        tmp_path,
        {"name": "test", "version": "1"},
        records=(
            _record(tmp_path, "transformed_image", "inputs/image.npy"),
            _record(tmp_path, "bone_segmentation", "inputs/bone.npy"),
            _record(tmp_path, "periosteal_mask", "inputs/peri.npy"),
            _record(tmp_path, "trabecular_mask", "inputs/trab.npy"),
        ),
    )
    write_manifest(manifest, tmp_path / "derivatives/Segmentation/manifest.json")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bone_microarchitecture.cli",
            "run-batch",
            str(tmp_path),
            "--spacing",
            "1",
            "1",
            "1",
            "--thickness-method",
            "edt",
            "--thickness-backend",
            "cpu",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )

    assert (tmp_path / "derivatives" / "Microarchitecture" / "manifest.json").exists()


def test_batch_filters_cases_by_subject_site_and_session(tmp_path):
    from bone_microarchitecture.batch import run_microarchitecture_batch

    _write_case(tmp_path, subject_id="SAMPLE001", site="radius", session_id="1")
    _write_case(tmp_path, subject_id="SAMPLE001", site="radius", session_id="2")
    _write_case(tmp_path, subject_id="SAMPLE001", site="tibia", session_id="1")

    records = run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        subject_id="SAMPLE001",
        site="radius",
        session_id="2",
        thickness_method="edt",
        thickness_backend="cpu",
    )

    measurement_records = [record for record in records if record.role == "measurements_table"]
    assert len(measurement_records) == 1
    assert measurement_records[0].subject_id == "SAMPLE001"
    assert measurement_records[0].site == "radius"
    assert measurement_records[0].session_id == "2"


def test_batch_filters_accept_site_alias_from_slicer_ui(tmp_path):
    """UI aliases such as radius_left and 00 must match parsed STRAMBO cases."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-STRAMBO_0001" / "site-radius" / "ses-Y00" / "masks"
    mask_dir.mkdir(parents=True)
    for role in ("seg", "full", "trab"):
        np.save(mask_dir / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)

    cases = batch._filter_cases(batch._discover_cases(tmp_path), subject_id="STRAMBO_0001", site="radius_left", session_id="00")

    assert len(cases) == 1
    assert cases[0]["bone_segmentation"].site == "radiusleft"


def test_batch_discovers_mids_style_raw_and_prefers_imported_contours(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture import batch

    root = tmp_path / "dataset"
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = root / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    for family in ("BoneContours", "ImportedContours"):
        for role in ("seg", "full", "trab"):
            output = (
                root
                / "derivatives"
                / family
                / "sub-001"
                / "ses-001"
                / "xct"
                / f"sub-001_ses-001_voi-radiusleft_desc-{role}_mask.nii.gz"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))

    cases = batch._discover_cases(root)

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path == raw
    assert "/derivatives/ImportedContours/" in str(cases[0]["bone_segmentation"].path)
    assert "/derivatives/ImportedContours/" in str(cases[0]["periosteal_mask"].path)
    assert "/derivatives/ImportedContours/" in str(cases[0]["trabecular_mask"].path)


def test_batch_filter_accepts_compact_voi_token_for_normalized_site(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture import batch

    root = tmp_path / "dataset"
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = root / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    for role in ("seg", "full", "trab"):
        output = (
            root
            / "derivatives"
            / "BoneContours"
            / "sub-001"
            / "ses-003"
            / "xct"
            / f"sub-001_ses-003_voi-radiusleft_desc-{role}_mask.nii.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))

    cases = batch._filter_cases(batch._discover_cases(root), subject_id="001", site="radiusleft", session_id="003")

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path == raw


def test_batch_discovers_bone_contours_with_canonical_sidecar_roles(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture import batch

    root = tmp_path / "dataset"
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = root / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    roles = {
        "seg": "bone_segmentation",
        "full": "periosteal_mask",
        "trab": "trabecular_mask",
    }
    for short_role, canonical_role in roles.items():
        output = (
            root
            / "derivatives"
            / "BoneContours"
            / "sub-001"
            / "ses-003"
            / "xct"
            / f"sub-001_ses-003_voi-radiusleft_desc-{short_role}_mask.nii.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))
        output.with_suffix("").with_suffix(".json").write_text(
            json.dumps({"role": canonical_role, "short_role": short_role}),
            encoding="utf-8",
        )

    cases = batch._filter_cases(batch._discover_cases(root), subject_id="001", site="radiusleft", session_id="003")

    assert len(cases) == 1
    assert cases[0]["bone_segmentation"].role == "bone_segmentation"
    assert cases[0]["periosteal_mask"].role == "periosteal_mask"
    assert cases[0]["trabecular_mask"].role == "trabecular_mask"


def test_batch_groups_compact_and_underscored_sites_into_one_case(tmp_path):
    from bone_microarchitecture import batch

    image = tmp_path / "image.npy"
    bone = tmp_path / "bone.npy"
    peri = tmp_path / "peri.npy"
    trab = tmp_path / "trab.npy"
    for path in (image, bone, peri, trab):
        np.save(path, np.ones((3, 3, 3), dtype=np.uint8))
    write_manifest(
        DerivativeManifest.create(
            "Segmentation",
            tmp_path,
            {"name": "test", "version": "1"},
            records=(
                replace(_record(tmp_path, "transformed_image", image, derivative="Registration"), site="radius_left", session_id="003"),
                replace(_record(tmp_path, "bone_segmentation", bone), site="radiusleft", session_id="003"),
                replace(_record(tmp_path, "periosteal_mask", peri), site="radiusleft", session_id="003"),
                replace(_record(tmp_path, "trabecular_mask", trab), site="radiusleft", session_id="003"),
            ),
        ),
        tmp_path / "derivatives/Segmentation/manifest.json",
    )

    cases = batch._filter_cases(batch._discover_cases(tmp_path), subject_id="SAMPLE001", site="radiusleft", session_id="003")

    assert len(cases) == 1
    assert sorted(cases[0]) == ["bone_segmentation", "periosteal_mask", "trabecular_mask", "transformed_image"]


def test_batch_common_region_mode_requires_native_common_region(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture import batch

    root = tmp_path / "dataset"
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = root / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    for role in ("seg", "full", "trab"):
        output = (
            root
            / "derivatives"
            / "BoneContours"
            / "sub-001"
            / "ses-003"
            / "xct"
            / f"sub-001_ses-003_voi-radiusleft_desc-{role}_mask.nii.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))

    with pytest.raises(ValueError, match=r"scan_region_native_common"):
        batch.run_microarchitecture_batch(
            root,
            use_common_region=True,
            require_common_region=True,
            subject_id="001",
            site="radiusleft",
            session_id="003",
        )


def test_batch_common_region_unstacked_matches_unstacked_case(tmp_path):
    from bone_microarchitecture import batch

    sitk = pytest.importorskip("SimpleITK")
    root = tmp_path / "dataset"
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = root / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    for role in ("seg", "full", "trab"):
        output = (
            root
            / "derivatives"
            / "BoneContours"
            / "sub-001"
            / "ses-003"
            / "xct"
            / f"sub-001_ses-003_voi-radiusleft_desc-{role}_mask.nii.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))
    common_path = (
        root
        / "derivatives"
        / "CommonRegion"
        / "sub-001"
        / "ses-003"
        / "xct"
        / "masks"
        / "sub-001_ses-003_voi-radiusleft_mask-scan-region_native_common.nii.gz"
    )
    common_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(common_path))
    write_manifest(
        DerivativeManifest.create(
            "CommonRegion",
            root,
            {"name": "test", "version": "1"},
            records=(
                DerivativeRecord(
                    "CommonRegion",
                    "scan_region_native_common",
                    "001",
                    "radiusleft",
                    "003",
                    None,
                    "native",
                    common_path,
                    "generated",
                    content_type="mask",
                ),
            ),
        ),
        root / "derivatives" / "CommonRegion" / "manifest.json",
    )

    records = batch.run_microarchitecture_batch(
        root,
        use_common_region=True,
        require_common_region=True,
        subject_id="001",
        site="radiusleft",
        session_id="003",
        thickness_method="edt",
        thickness_backend="cpu",
    )

    assert records
    assert records[0].metadata["use_common_region"] is True


def test_batch_common_region_mode_rejects_stack_one_common_region_for_unstacked_case(tmp_path):
    from bone_microarchitecture import batch

    sitk = pytest.importorskip("SimpleITK")
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    raw = tmp_path / "sub-001" / "ses-003" / "xct" / "sub-001_ses-003_voi-radiusleft_xct.nii.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(raw))
    for role in ("seg", "full", "trab"):
        output = (
            tmp_path
            / "derivatives"
            / "BoneContours"
            / "sub-001"
            / "ses-003"
            / "xct"
            / f"sub-001_ses-003_voi-radiusleft_desc-{role}_mask.nii.gz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(output))
    common_path = (
        tmp_path
        / "derivatives"
        / "CommonRegion"
        / "sub-001"
        / "ses-003"
        / "xct"
        / "masks"
        / "sub-001_ses-003_voi-radiusleft_stack-01_mask-scan-region_native_common.nii.gz"
    )
    common_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(common_path))
    write_manifest(
        DerivativeManifest.create(
            "CommonRegion",
            tmp_path,
            {"name": "test", "version": "1"},
            records=(
                DerivativeRecord(
                    "CommonRegion",
                    "scan_region_native_common",
                    "001",
                    "radiusleft",
                    "003",
                    1,
                    "native",
                    common_path,
                    "generated",
                    content_type="mask",
                ),
            ),
        ),
        tmp_path / "derivatives" / "CommonRegion" / "manifest.json",
    )

    with pytest.raises(ValueError, match=r"scan_region_native_common"):
        batch.run_microarchitecture_batch(
            tmp_path,
            use_common_region=True,
            require_common_region=True,
            subject_id="001",
            site="radiusleft",
            session_id="003",
            thickness_method="edt",
            thickness_backend="cpu",
        )


def test_batch_refuses_incomplete_rows_before_starting_measurements(tmp_path, monkeypatch):
    from bone_microarchitecture import batch

    sitk = pytest.importorskip("SimpleITK")
    image = np.ones((3, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    image_path = tmp_path / "sub-001" / "ses-001" / "xct" / "sub-001_ses-001_voi-radiusleft_xct.nii.gz"
    image_path.parent.mkdir(parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
    for role in ("seg", "full"):
        path = (
            tmp_path
            / "derivatives"
            / "ImportedContours"
            / "sub-001"
            / "ses-001"
            / "xct"
            / f"sub-001_ses-001_voi-radiusleft_desc-{role}_mask.npy"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)

    monkeypatch.setattr(batch, "compute_microarchitecture", lambda **_kwargs: pytest.fail("must not run"))

    with pytest.raises(ValueError, match=r"sub-001.*ses-001.*radiusleft.*trabecular_mask"):
        batch.run_microarchitecture_batch(
            tmp_path,
            spacing=(1.0, 1.0, 1.0),
            thickness_method="edt",
            thickness_backend="cpu",
        )


def test_batch_does_not_treat_existing_microarchitecture_maps_as_incomplete_inputs(tmp_path):
    from bone_microarchitecture.batch import run_microarchitecture_batch

    _write_case(tmp_path)
    map_path = (
        tmp_path
        / "derivatives"
        / "Microarchitecture"
        / "sub-002"
        / "ses-001"
        / "xct"
        / "maps"
        / "sub-002_ses-001_voi-radiusleft_map-tb-th.npy"
    )
    map_path.parent.mkdir(parents=True)
    np.save(map_path, np.ones((3, 3, 3), dtype=np.float32))
    write_manifest(
        DerivativeManifest.create(
            "Microarchitecture",
            tmp_path,
            {"name": "test", "version": "1"},
            records=(
                DerivativeRecord(
                    derivative="Microarchitecture",
                    role="trabecular_thickness_map",
                    subject_id="002",
                    site="radius_left",
                    session_id="001",
                    stack_index=None,
                    space="native",
                    path=map_path,
                    source="generated",
                    content_type="image",
                ),
            ),
        ),
        tmp_path / "derivatives" / "Microarchitecture" / "manifest.json",
    )

    records = run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        thickness_method="edt",
        thickness_backend="cpu",
    )

    assert len(records) == 1
    assert records[0].subject_id == "SAMPLE001"


def test_run_batch_treats_selected_derivatives_folder_as_dataset_root(tmp_path):
    """Selecting derivatives in a GUI must not turn outputs into derivatives/derivatives."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-STRAMBO_0001" / "site-radius" / "ses-Y00" / "masks"
    mask_dir.mkdir(parents=True)
    for role in ("seg", "full", "trab"):
        np.save(mask_dir / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)

    records = run_microarchitecture_batch(
        tmp_path / "derivatives",
        spacing=(1.0, 1.0, 1.0),
        subject_id="STRAMBO_0001",
        site="radius_left",
        session_id="00",
        thickness_method="edt",
        thickness_backend="cpu",
    )

    assert records
    assert (tmp_path / "derivatives" / "Microarchitecture" / "manifest.json").is_file()
    assert not (tmp_path / "derivatives" / "derivatives").exists()


def test_cli_run_batch_accepts_case_filters(tmp_path, monkeypatch):
    from bone_microarchitecture import cli

    received = {}

    def fake_run_microarchitecture_batch(dataset_root, **kwargs):
        received.update(dataset_root=dataset_root, **kwargs)
        return []

    monkeypatch.setattr(cli, "run_microarchitecture_batch", fake_run_microarchitecture_batch)

    assert cli.main([
        "run-batch",
        str(tmp_path),
        "--subject",
        "SAMPLE001",
        "--site",
        "radius",
        "--session",
        "2",
    ]) == 0
    assert received["subject_id"] == "SAMPLE001"
    assert received["site"] == "radius"
    assert received["session_id"] == "2"


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


def test_batch_discovers_root_images_with_derivative_segmentation_masks(tmp_path):
    """Microarchitecture batch should pair root scans with Bone Contouring masks in derivatives."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-STRAMBO_0001" / "site-radius" / "ses-Y00" / "masks"
    mask_dir.mkdir(parents=True)
    for role in ("seg", "full", "trab", "cort"):
        np.save(mask_dir / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path.name == "STRAMBO_0001_RL_Y00_image.npy"
    assert cases[0]["bone_segmentation"].path.name == "STRAMBO_0001_RL_Y00_mask-seg.npy"
    assert cases[0]["periosteal_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-full.npy"
    assert cases[0]["trabecular_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-trab.npy"
    assert cases[0]["cortical_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-cort.npy"


def test_batch_discovery_ignores_timelapse_outputs_as_individual_inputs(tmp_path):
    """Timelapse fused outputs should not duplicate native microarchitecture cases."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    for role in ("seg", "full", "trab"):
        np.save(tmp_path / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)
    fused_dir = tmp_path / "derivatives" / "Timelapse" / "sub-STRAMBO_0001" / "ses-00" / "xct" / "images"
    fused_dir.mkdir(parents=True)
    fused_image = fused_dir / "sub-STRAMBO_0001_site-radius_left_ses-00_image_fused.npy"
    fused_seg = fused_dir / "sub-STRAMBO_0001_site-radius_left_ses-00_stack-01_seg.npy"
    np.save(fused_image, image)
    np.save(fused_seg, mask)
    write_manifest(
        DerivativeManifest.create(
            "Timelapse",
            tmp_path,
            {"name": "timelapsed", "version": "test"},
            records=(
                DerivativeRecord(
                    "Timelapse",
                    "transformed_image",
                    "STRAMBO_0001",
                    "radius_left",
                    "00",
                    1,
                    "native",
                    fused_image,
                    "generated",
                    content_type="image",
                ),
                DerivativeRecord(
                    "Timelapse",
                    "bone_segmentation",
                    "STRAMBO_0001",
                    "radius_left",
                    "00",
                    1,
                    "native",
                    fused_seg,
                    "generated",
                    content_type="mask",
                ),
            ),
        ),
        tmp_path / "derivatives" / "Timelapsed" / "manifest.json",
    )

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path.name == "STRAMBO_0001_RL_Y00_image.npy"


def test_batch_discovery_does_not_use_measurement_maps_as_grayscale_inputs(tmp_path):
    """Previously written microarchitecture maps are outputs, not input images."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    for role in ("seg", "full", "trab"):
        np.save(tmp_path / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)
    map_path = tmp_path / "derivatives" / "Microarchitecture" / "sub-STRAMBO_0001" / "site-radius_left" / "native_space" / "ses-00" / "maps" / "tb-th.npy"
    map_path.parent.mkdir(parents=True)
    np.save(map_path, image)
    write_manifest(
        DerivativeManifest.create(
            "Microarchitecture",
            tmp_path,
            {"name": "bone-microarchitecture", "version": "test"},
            records=(
                DerivativeRecord(
                    "Microarchitecture",
                    "trabecular_thickness_map",
                    "STRAMBO_0001",
                    "radius_left",
                    "00",
                    None,
                    "native",
                    map_path,
                    "generated",
                    content_type="image",
                ),
            ),
        ),
        tmp_path / "derivatives" / "Microarchitecture" / "manifest.json",
    )

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path.name == "STRAMBO_0001_RL_Y00_image.npy"


def test_batch_discovers_scanco_style_masks_beside_root_scan(tmp_path):
    """Native Scanco mask exports beside the image should be valid batch inputs too."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "STRAMBO_0001_RL_Y00_image.npy", image)
    for role in ("seg", "full", "trab"):
        np.save(tmp_path / f"STRAMBO_0001_RL_Y00_mask-{role}.npy", mask)

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0]["bone_segmentation"].path.name == "STRAMBO_0001_RL_Y00_mask-seg.npy"
    assert cases[0]["periosteal_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-full.npy"
    assert cases[0]["trabecular_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-trab.npy"


def test_batch_discovers_sidecar_described_non_aim_inputs(tmp_path):
    """Shared discovery lets non-AIM files participate when metadata lives in sidecars."""
    from bone_microarchitecture import batch

    image = tmp_path / "baseline_scan.nii.gz"
    image.write_bytes(b"nifti")
    image.with_name("baseline_scan.json").write_text(
        json.dumps({"subject_id": "S01", "session_id": "baseline", "site": "tibia_right"})
    )
    mask_dir = tmp_path / "derivatives" / "Segmentation" / "unstructured"
    mask_dir.mkdir(parents=True)
    for name, role in {
        "mineralized.nii.gz": "segmentation",
        "outer-roi.nii.gz": "full",
        "inner-roi.nii.gz": "trab",
    }.items():
        path = mask_dir / name
        path.write_bytes(b"mask")
        path.with_name(f"{name}.json").write_text(
            json.dumps({
                "subject_id": "S01",
                "session_id": "baseline",
                "site": "tibia_right",
                "role": role,
            })
        )

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    case = cases[0]
    assert case["transformed_image"].path.name == "baseline_scan.nii.gz"
    assert case["bone_segmentation"].path.name == "mineralized.nii.gz"
    assert case["periosteal_mask"].path.name == "outer-roi.nii.gz"
    assert case["trabecular_mask"].path.name == "inner-roi.nii.gz"
    assert case["bone_segmentation"].subject_id == "S01"
    assert case["bone_segmentation"].session_id == "baseline"
    assert case["bone_segmentation"].site == "tibiaright"


def test_batch_discovers_bare_scanco_aim_scan_with_aim_masks(tmp_path):
    """Bare STRAMBO AIM names should pair with root Scanco masks without a manifest."""
    from bone_microarchitecture import batch

    (tmp_path / "STRAMBO_0001_RL_Y00.AIM").write_bytes(b"aim")
    for role in ("seg", "full", "trab"):
        (tmp_path / f"STRAMBO_0001_RL_Y00_mask-{role}.AIM").write_bytes(b"aim-mask")

    cases = batch._discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0]["transformed_image"].path.name == "STRAMBO_0001_RL_Y00.AIM"
    assert cases[0]["transformed_image"].site == "radiusleft"
    assert cases[0]["transformed_image"].session_id == "00"
    assert cases[0]["bone_segmentation"].path.name == "STRAMBO_0001_RL_Y00_mask-seg.AIM"
    assert cases[0]["periosteal_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-full.AIM"
    assert cases[0]["trabecular_mask"].path.name == "STRAMBO_0001_RL_Y00_mask-trab.AIM"


def test_batch_discovery_preserves_left_and_right_site_identity(tmp_path):
    """Left and right scans must not collapse into one generic radius case."""
    from bone_microarchitecture import batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    for side in ("RL", "RR"):
        np.save(tmp_path / f"STRAMBO_0001_{side}_Y00_image.npy", image)
        mask_dir = tmp_path / "derivatives" / "Segmentation" / "sub-STRAMBO_0001" / "site-radius" / "ses-Y00" / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for role in ("seg", "full", "trab"):
            np.save(mask_dir / f"STRAMBO_0001_{side}_Y00_mask-{role}.npy", mask)

    cases = batch._discover_cases(tmp_path)

    assert sorted(case["bone_segmentation"].site for case in cases) == ["radiusleft", "radiusright"]


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


def test_registered_and_native_measurements_share_native_maps(tmp_path, monkeypatch):
    """Common-region measurements should reuse native maps instead of recomputing them."""
    import bone_microarchitecture.batch as batch

    image = np.full((4, 4, 4), 100.0, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    common_region = np.zeros((4, 4, 4), dtype=np.uint8)
    common_region[:2, :, :] = 1
    paths = {
        "image": "inputs/image.npy",
        "bone": "inputs/bone.npy",
        "peri": "inputs/peri.npy",
        "trab": "inputs/trab.npy",
        "common": "inputs/common.npy",
    }
    for name, array in {
        "image": image,
        "bone": mask,
        "peri": mask,
        "trab": mask,
        "common": common_region,
    }.items():
        path = tmp_path / paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
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
                _record(tmp_path, "scan_region_native_common", paths["common"], derivative="CommonRegion"),
            ),
        ),
        tmp_path / "derivatives/Segmentation/manifest.json",
    )

    batch.run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        thickness_method="edt",
        thickness_backend="cpu",
        require_common_region=True,
    )
    registered_records = read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
    assert any("/xct/registered_measurements/" in str(record.path) for record in registered_records)
    assert any("/xct/maps/" in str(record.path) for record in registered_records)
    assert not any("/xct/registered/maps/" in str(record.path) for record in registered_records)

    def should_not_compute(**_kwargs):
        raise AssertionError("native maps should be reused")

    monkeypatch.setattr(batch, "compute_microarchitecture", should_not_compute)
    progress_events = []
    batch.run_microarchitecture_batch(
        tmp_path,
        spacing=(1.0, 1.0, 1.0),
        thickness_method="edt",
        thickness_backend="cpu",
        use_common_region=False,
        progress=progress_events.append,
    )
    records = read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
    assert any("/xct/measurements/" in str(record.path) for record in records)
    assert sum(1 for record in records if record.role == "trabecular_thickness_map") == 1
    assert any(
        event.step == "maps"
        and event.status == "reused"
        and event.message == "Reused native microarchitecture maps"
        and str(event.path).endswith("/xct/maps")
        for event in progress_events
    )


def test_grayscale_batch_persists_bmd_maps_with_deterministic_generic_role(tmp_path):
    """Every BMD map returned by the core pipeline must be discoverable in the manifest."""
    from bone_microarchitecture.batch import run_microarchitecture_batch

    image = np.full((4, 4, 4), 250.0, dtype=np.float32)
    for name in ("image", "bone_mask", "periosteal_mask", "trabecular_mask", "cortical_mask"):
        np.save(tmp_path / f"{name}.npy", image)

    run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")

    bmd_records = [
        record for record in read_manifest(tmp_path / "derivatives/Microarchitecture/manifest.json").records
        if record.role == "material_map" and record.metadata.get("map_name") in {"Tt.BMD", "Tb.BMD", "Ct.BMD"}
    ]
    assert {record.metadata["map_name"] for record in bmd_records} == {"Tt.BMD", "Tb.BMD", "Ct.BMD"}
    assert all(record.path.is_file() for record in bmd_records)


def test_batch_recomputes_compatible_outputs_when_ttbmd_is_missing_from_old_csv(tmp_path, monkeypatch):
    """Old reused measurements without Tt.BMD should not hide current required rows."""
    from bone_microarchitecture import batch

    _write_case(tmp_path)
    calls = []
    original_compute = batch.compute_microarchitecture

    def fake_compute(**kwargs):
        calls.append(kwargs)
        return original_compute(**kwargs)

    monkeypatch.setattr(batch, "compute_microarchitecture", fake_compute)
    records = batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")
    assert len(calls) == 1

    csv_path = records[0].path
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Parameter"] != "Tt.BMD":
                rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    calls.clear()
    batch.run_microarchitecture_batch(tmp_path, spacing=(1.0, 1.0, 1.0), thickness_method="edt", thickness_backend="cpu")

    assert len(calls) == 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert "Tt.BMD" in {row["Parameter"] for row in csv.DictReader(handle)}

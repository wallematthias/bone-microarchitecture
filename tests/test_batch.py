from __future__ import annotations

import csv

import numpy as np

from bone_imaging_derivatives import DerivativeManifest, DerivativeRecord, write_manifest


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

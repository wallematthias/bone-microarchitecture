"""Definition and topology regressions, not claims of bitwise IPL parity."""

import numpy as np
import pytest

from bone_microarchitecture import compute_microarchitecture
from bone_microarchitecture.batch import _summarize_measurements
from bone_microarchitecture.results import measurement_rows
from bone_microarchitecture.thickness import hildebrand_thickness_map


def test_number_export_uses_inverse_mean_spacing_not_mean_inverse():
    # Spacings 0.5 and 1.5 mm have mean 1 mm, hence Tb.N = 1/mm.
    maps = {"Tb.N": np.array([[[2.0, 2.0 / 3.0]]])}
    row = measurement_rows({"Tb.N": 1.0}, maps)[0]
    assert row["Mean"] == pytest.approx(1.0)
    assert row["Median"] == pytest.approx(4.0 / 3.0)


def test_batch_number_uses_inverse_mean_spacing():
    domain = np.ones((1, 1, 2), dtype=bool)
    masks = {"bone_segmentation": domain, "periosteal_mask": domain, "trabecular_mask": domain}
    maps = {"Tb.N": np.array([[[2.0, 2.0 / 3.0]]]), "Tb.Th": domain.astype(float), "Tb.Sp": np.zeros(domain.shape)}
    result = _summarize_measurements(np.zeros(domain.shape), masks, maps, (1.0, 1.0, 1.0))
    assert result["Tb.N"] == pytest.approx(1.0)
    assert result["Tb.1/N.SD"] == pytest.approx(0.5)


def test_cortical_thickness_is_independent_of_mineralized_pores():
    cort = np.zeros((11, 11, 11), dtype=bool)
    cort[2:9, 2:9, 2:9] = True
    porous_bone = cort.copy()
    porous_bone[4:7, 4:7, 4:7] = False
    args = dict(periosteal_mask=cort, trabecular_mask=np.zeros_like(cort), cortical_mask=cort,
                spacing=(0.1, 0.1, 0.1), thickness_backend="cpu")
    solid = compute_microarchitecture(bone_mask=cort, **args)
    porous = compute_microarchitecture(bone_mask=porous_bone, **args)
    np.testing.assert_array_equal(porous.maps["Ct.Th"], solid.maps["Ct.Th"])
    assert porous.measurements["Ct.Th"] == pytest.approx(solid.measurements["Ct.Th"])


def test_incomplete_sphere_centers_at_image_end_are_suppressed():
    from bone_microarchitecture.thickness import _accumulate_local_diameters_cpu
    distance = np.zeros((9, 9, 9), dtype=float)
    distance[0, 4, 4] = 2.0  # This sphere extends beyond the available scan.
    centers = distance > 0
    result = _accumulate_local_diameters_cpu(distance, centers, (1., 1., 1.), 0.5)
    assert not result.any()


def test_complete_spheres_are_not_removed_just_because_output_reaches_end_slice():
    from bone_microarchitecture.thickness import _accumulate_local_diameters_cpu
    distance = np.zeros((9, 9, 9), dtype=float)
    distance[1, 4, 4] = 1.0
    result = _accumulate_local_diameters_cpu(distance, distance > 0, (1., 1., 1.), 0.5)
    assert result[0, 4, 4] == pytest.approx(1.0)


def test_spacing_uses_bone_context_outside_reporting_domain():
    bone = np.zeros((11, 11, 11), dtype=bool)
    bone[2:9, 2:9, 2] = True
    bone[2:9, 2:9, 8] = True
    domain = np.zeros_like(bone)
    domain[3:8, 3:8, 4:7] = True
    result = compute_microarchitecture(bone_mask=bone, periosteal_mask=np.ones_like(bone),
                                      trabecular_mask=domain, spacing=(1., 1., 1.),
                                      thickness_backend="cpu")
    # The 3-voxel ROI is only a reporting boundary, not a marrow wall.
    assert result.maps["Tb.Sp"][5, 5, 5] > 3.0


def test_cortical_pores_exclude_unseeded_surface_erosion_and_tiny_noise():
    cort = np.zeros((9, 41, 41), dtype=bool)
    cort[:, 3:38, 3:38] = True
    cort[:, 12:29, 12:29] = False
    bone = cort.copy()
    bone[2:7, 7, 7] = False  # Five-voxel longitudinal canal, surrounded in XY.
    bone[4, 9, 9] = False   # Isolated one-voxel noise (not connected to the canal).
    bone[:, 3:6, 18:22] = False  # Surface-connected erosion, never seeded.
    result = compute_microarchitecture(bone_mask=bone, periosteal_mask=np.ones_like(cort),
                                      trabecular_mask=np.zeros_like(cort), cortical_mask=cort,
                                      spacing=(1., 1., 1.), thickness_method="edt", thickness_backend="cpu")
    assert result.measurements["Ct.Po.V"] == pytest.approx(5.0)
    assert result.measurements["Ct.Po"] == pytest.approx(5.0 / 8424.0)
    assert not result.maps["Ct.Po.Dm"][:, 3:6, 18:22].any()


def test_registered_reporting_does_not_repeat_pore_cleanup_after_cropping():
    from bone_microarchitecture.batch import restrict_measurements
    cort = np.zeros((9, 41, 41), dtype=bool)
    cort[:, 3:38, 3:38] = True
    cort[:, 12:29, 12:29] = False
    bone = cort.copy()
    bone[2:7, 7, 7] = False
    masks = {"bone_segmentation": bone, "periosteal_mask": np.ones_like(bone),
             "trabecular_mask": np.zeros_like(bone), "cortical_mask": cort}
    native = compute_microarchitecture(bone_mask=bone, periosteal_mask=masks["periosteal_mask"],
                                      trabecular_mask=masks["trabecular_mask"], cortical_mask=cort,
                                      spacing=(1., 1., 1.), thickness_backend="cpu")
    region = np.zeros_like(bone)
    region[4] = True
    restricted = restrict_measurements(native, masks, region, (1., 1., 1.))
    assert restricted.measurements["Ct.Po.V"] == 1.0
    assert restricted.maps["Ct.Po.Mask"].sum() == 1
    np.testing.assert_array_equal(restricted.maps["Ct.Th"][4], native.maps["Ct.Th"][4])
    assert not restricted.maps["Ct.Th"][:4].any()
    assert "Ct.BMD" not in restricted.measurements


def test_csv_records_scientific_method_separately_from_package_version(tmp_path):
    import json
    from bone_microarchitecture.method import METHOD_ID
    from bone_microarchitecture.results import write_measurement_csv
    path = tmp_path / "measurements.csv"
    write_measurement_csv(path, {"Ct.Po": 0.02})
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["measurement_method"] == METHOD_ID
    assert metadata["ipl_parity"] == "not-validated"


@pytest.mark.parametrize("length,expected", [(4, 0), (5, 5), (9, 9)])
def test_pore_cleanup_retains_open_z_canals_but_removes_under_five_voxels(length, expected):
    from bone_microarchitecture.pores import cortical_pore_mask
    cort = np.zeros((9, 41, 41), dtype=bool)
    cort[:, 3:38, 3:38] = True
    cort[:, 12:29, 12:29] = False
    bone = cort.copy()
    bone[:length, 7, 7] = False
    pores = cortical_pore_mask(bone, cort)
    assert pores.sum() == expected
    np.testing.assert_array_equal(pores, np.flip(cortical_pore_mask(np.flip(bone, 0), np.flip(cort, 0)), 0))


def test_sphere_eligibility_is_symmetric_and_uses_physical_spacing():
    from bone_microarchitecture.thickness import _accumulate_local_diameters_cpu
    distance = np.zeros((9, 9, 9), dtype=float)
    distance[0, 4, 4] = distance[-1, 4, 4] = 0.25
    distance[1, 4, 4] = distance[-2, 4, 4] = 0.25
    result = _accumulate_local_diameters_cpu(distance, distance > 0, (0.2, 1., 1.), 0.5)
    np.testing.assert_array_equal(result, np.flip(result, 0))
    assert result[1, 4, 4] == pytest.approx(0.3)
    # End-slice centres are incomplete even though radius < one XY voxel.
    only_ends = distance.copy()
    only_ends[1:-1] = 0
    assert not _accumulate_local_diameters_cpu(only_ends, only_ends > 0, (0.2, 1., 1.), 0.5).any()


def test_scientific_method_invalidates_pre_alignment_cache_hash(monkeypatch):
    from bone_microarchitecture import batch
    args = ((), (0.0607, 0.0607, 0.0607), False, "hildebrand", "cpu")
    new_hash = batch._compatibility_hash(*args)
    monkeypatch.setattr(batch, "METHOD_METADATA", {})
    assert batch._compatibility_hash(*args) != new_hash


def test_restricting_existing_density_maps_does_not_replace_bmd_with_zero():
    from bone_microarchitecture.batch import restrict_measurements
    peri = np.zeros((7, 7, 7), dtype=bool)
    peri[1:6, 1:6, 1:6] = True
    cort = np.zeros_like(peri)
    cort[2:5, 2:5, 2:5] = True
    trab = peri & ~cort
    masks = {"bone_segmentation": peri, "periosteal_mask": peri,
             "trabecular_mask": trab, "cortical_mask": cort}
    native = compute_microarchitecture(bone_mask=peri, periosteal_mask=peri,
                                      trabecular_mask=trab, cortical_mask=cort,
                                      grayscale=np.full(peri.shape, 100., dtype=np.float32),
                                      spacing=(1., 1., 1.), thickness_backend="cpu")
    restricted = restrict_measurements(native, masks, peri, (1., 1., 1.))
    rows = {row["Parameter"]: row for row in measurement_rows(restricted.measurements, restricted.maps)}
    for name in ("Tt.BMD", "Tb.BMD", "Ct.BMD"):
        assert restricted.measurements[name] == pytest.approx(100.)
        assert rows[name]["Mean"] == restricted.measurements[name]

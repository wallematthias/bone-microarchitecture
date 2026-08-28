from __future__ import annotations

import csv

import numpy as np
import pytest

from bone_microarchitecture import compute_microarchitecture
from bone_microarchitecture.results import (
    MEASUREMENT_ORDER,
    PARAMETER_DEFINITIONS,
    measurement_rows,
    write_measurement_csv,
)
from bone_microarchitecture.thickness import _medial_axis, hildebrand_thickness_map, local_thickness_map


def test_compute_microarchitecture_returns_measurements_and_maps(tmp_path):
    peri = np.ones((7, 7, 7), dtype=bool)
    trab = np.zeros_like(peri)
    trab[2:5, 2:5, 2:5] = True
    trab_region = np.zeros_like(peri)
    trab_region[1:6, 1:6, 1:6] = True
    cort = peri & ~trab_region
    bone = trab | cort
    grayscale = np.full(peri.shape, 800.0, dtype=float)
    grayscale[trab] = 300.0
    grayscale[cort] = 900.0

    result = compute_microarchitecture(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab_region,
        cortical_mask=cort,
        grayscale=grayscale,
        spacing=(0.1, 0.1, 0.1),
    )

    assert result.measurements["Tb.BV"] == pytest.approx(0.027)
    assert result.measurements["Tb.TV"] == pytest.approx(0.125)
    assert result.measurements["Tb.BV/TV"] == pytest.approx(27 / 125)
    assert result.measurements["Tb.BMD"] == pytest.approx(((27 * 300.0) + (98 * 800.0)) / 125.0)
    assert result.measurements["Ct.BMD"] == pytest.approx(900.0)
    assert result.measurements["Ct.Po"] == pytest.approx(0.0)
    assert result.maps["Tb.Th"].shape == trab.shape
    assert result.maps["Tb.Sp"].shape == trab.shape
    assert result.maps["Ct.Th"].shape == trab.shape
    assert result.maps["Tb.BMD"].shape == trab.shape
    assert result.maps["Ct.BMD"].shape == trab.shape

    rows = measurement_rows(result.measurements, result.maps)
    row_by_parameter = {row["Parameter"]: row for row in rows}
    assert rows[0]["Parameter"] == "Tb.BMD"
    assert rows[1]["Parameter"] == "Tb.BV/TV"
    assert row_by_parameter["Tb.BV/TV"]["Units"] == "fraction"
    assert row_by_parameter["Tb.Th"]["Mean"] == pytest.approx(result.measurements["Tb.Th"])
    assert row_by_parameter["Tb.Th"]["Median"] != ""
    assert row_by_parameter["Tb.Th"]["P5"] != ""
    assert row_by_parameter["Tb.Th"]["P95"] != ""
    assert row_by_parameter["Tb.N"]["Mean"] == pytest.approx(result.measurements["Tb.N"])
    assert row_by_parameter["Tb.N"]["Median"] != ""
    assert row_by_parameter["Tb.N"]["SD"] != ""
    assert row_by_parameter["Tb.N"]["P5"] != ""
    assert result.maps["Tb.N"].shape == trab.shape
    assert np.count_nonzero(result.maps["Tb.N"]) > 0
    assert "Tb.1/N.SD" in result.measurements
    assert "Mean Tb.Th" not in result.measurements
    assert "Mean Tb.Sp" not in result.measurements
    csv_path = tmp_path / "measurements.csv"
    write_measurement_csv(csv_path, result.measurements, result.maps)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert exported[0]["Parameter"] == "Tb.BMD"
    assert exported[1]["Parameter"] == "Tb.BV/TV"
    assert "P95" in exported[1]


def test_reported_measurements_have_parameter_definitions():
    assert set(MEASUREMENT_ORDER) <= set(PARAMETER_DEFINITIONS)


def test_compartment_masks_are_intersected_with_bone_segmentation_for_bone_measures_but_not_bmd():
    peri = np.zeros((5, 5, 5), dtype=bool)
    peri[1:4, 1:4, 1:4] = True
    trab_compartment = np.zeros_like(peri)
    trab_compartment[1:4, 1:4, 1:3] = True
    cort_compartment = peri & ~trab_compartment
    bone = np.zeros_like(peri)
    bone[2, 2, 1] = True
    bone[2, 2, 3] = True
    grayscale = np.zeros(peri.shape, dtype=np.float32)
    grayscale[trab_compartment] = 100.0
    grayscale[cort_compartment] = 200.0
    grayscale[bone & trab_compartment] = 300.0
    grayscale[bone & cort_compartment] = 900.0

    result = compute_microarchitecture(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab_compartment,
        cortical_mask=cort_compartment,
        grayscale=grayscale,
        spacing=(1.0, 1.0, 1.0),
    )

    assert result.measurements["Tb.BV"] == pytest.approx(1.0)
    assert result.measurements["Tb.TV"] == pytest.approx(18.0)
    assert result.measurements["Ct.BV"] == pytest.approx(1.0)
    assert result.measurements["Ct.TV"] == pytest.approx(9.0)
    assert result.measurements["Tb.BMD"] == pytest.approx(((17 * 100.0) + 300.0) / 18.0)
    assert result.measurements["Ct.BMD"] == pytest.approx(((8 * 200.0) + 900.0) / 9.0)
    assert np.count_nonzero(result.maps["Tb.Th"]) == 1
    assert np.count_nonzero(result.maps["Ct.Th"]) == 1
    assert np.count_nonzero(result.maps["Tb.BMD"]) == 18
    assert np.count_nonzero(result.maps["Ct.BMD"]) == 9
    assert result.maps["Tb.Th"][2, 2, 1] > 0
    assert result.maps["Ct.Th"][2, 2, 3] > 0


def test_cortical_porosity_reports_pore_diameter_distribution():
    peri = np.ones((9, 9, 9), dtype=bool)
    trab = np.zeros_like(peri)
    trab[3:6, 3:6, 3:6] = True
    cort = peri & ~trab
    bone = cort.copy()
    bone[1:4, 1:4, 1:4] = False
    bone[5:8, 5:8, 5:8] = False

    result = compute_microarchitecture(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab,
        cortical_mask=cort,
        spacing=(0.1, 0.1, 0.1),
        thickness_backend="cpu",
    )

    rows = {row["Parameter"]: row for row in measurement_rows(result.measurements, result.maps)}

    assert result.measurements["Ct.Po"] > 0
    assert "Ct.Po.Dm" in result.measurements
    assert result.maps["Ct.Po.Dm"].shape == cort.shape
    assert np.count_nonzero(result.maps["Ct.Po.Dm"]) > 0
    assert rows["Ct.Po"]["Median"] == ""
    assert rows["Ct.Po.Dm"]["Median"] != ""
    assert rows["Ct.Po.Dm"]["SD"] != ""


def test_shape_mismatch_raises_clear_error():
    mask = np.ones((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="same shape"):
        compute_microarchitecture(
            periosteal_mask=mask,
            trabecular_mask=np.ones((3, 3, 2), dtype=bool),
            spacing=(0.1, 0.1, 0.1),
        )


def test_spacing_must_be_three_positive_values():
    mask = np.ones((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="three positive"):
        compute_microarchitecture(
            periosteal_mask=mask,
            trabecular_mask=mask,
            spacing=(0.1, 0.0, 0.1),
        )


def test_thickness_map_uses_bounded_distance_transform_memory():
    import inspect

    from bone_microarchitecture import thickness

    source = inspect.getsource(thickness.local_thickness_map)
    assert "distance_transform_edt" in source
    assert "binary_dilation" not in source
    assert "_ellipsoid_structure" not in source


def test_exact_hildebrand_thickness_has_cpu_and_optional_mps_backends():
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[2:5, 2:5, 2:5] = True

    bounded = local_thickness_map(mask, (1.0, 1.0, 1.0))
    exact = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="cpu")

    assert exact.shape == mask.shape
    assert exact.dtype == np.float32
    assert np.count_nonzero(exact[~mask]) == 0
    assert exact[3, 3, 3] >= bounded[3, 3, 3] - 1.0


def test_exact_hildebrand_mps_backend_uses_optional_metal_backend():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    metal = pytest.importorskip("bone_microarchitecture.metal")

    if not metal.is_metal_available():
        with pytest.raises(RuntimeError, match="Metal"):
            hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="mps")
    else:
        cpu = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="cpu")
        mps = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="mps")
        np.testing.assert_allclose(mps, cpu, rtol=1e-5, atol=1e-5)


def test_mps_backend_does_not_use_pytorch_tensor_loop():
    import inspect

    from bone_microarchitecture import thickness

    source = inspect.getsource(thickness._accumulate_local_diameters_mps)

    assert "metal_hildebrand_thickness_map" in source
    assert "torch" not in source


def test_metal_backend_defines_atomic_sphere_fitting_kernel():
    from bone_microarchitecture import metal

    assert "kernel void accumulate_local_diameters" in metal.METAL_SOURCE
    assert "atomic_fetch_max_explicit" in metal.METAL_SOURCE


def test_exact_hildebrand_opencl_backend_uses_optional_opencl_backend():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    opencl = pytest.importorskip("bone_microarchitecture.opencl")

    if not opencl.is_opencl_available():
        with pytest.raises(RuntimeError, match="OpenCL"):
            hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="opencl")
    else:
        cpu = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="cpu")
        gpu = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="opencl")
        np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-5)


def test_opencl_backend_defines_atomic_sphere_fitting_kernel():
    from bone_microarchitecture import opencl

    assert "accumulate_local_diameters_kernel" in opencl.OPENCL_SOURCE
    assert "atomic_max" in opencl.OPENCL_SOURCE


def test_pipeline_can_request_exact_thickness_backend():
    peri = np.ones((5, 5, 5), dtype=bool)
    trab = np.zeros_like(peri)
    trab[1:4, 1:4, 1:4] = True

    result = compute_microarchitecture(
        periosteal_mask=peri,
        trabecular_mask=trab,
        spacing=(1.0, 1.0, 1.0),
        thickness_method="hildebrand",
        thickness_backend="cpu",
    )

    assert result.maps["Tb.Th"].shape == trab.shape
    assert result.metadata["thickness_method"] == "hildebrand"
    assert result.metadata["thickness_backend"] == "cpu"


def test_pipeline_defaults_to_exact_sphere_fitting_with_auto_backend():
    peri = np.ones((5, 5, 5), dtype=bool)
    trab = np.zeros_like(peri)
    trab[1:4, 1:4, 1:4] = True

    result = compute_microarchitecture(
        periosteal_mask=peri,
        trabecular_mask=trab,
        spacing=(1.0, 1.0, 1.0),
    )

    assert result.metadata["thickness_method"] == "hildebrand"
    assert result.metadata["thickness_backend"] in {"cpu", "mps"}


def test_exact_hildebrand_auto_backend_resolves_to_platform_default():
    from bone_microarchitecture.thickness import default_thickness_backend

    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    result = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="auto")

    assert default_thickness_backend() in {"cpu", "mps", "opencl"}
    expected = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend=default_thickness_backend())
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_medial_axis_uses_dominated_center_pruning_only():
    from scipy import ndimage

    rng = np.random.default_rng(2)
    mask = ndimage.binary_closing(rng.random((12, 12, 12)) > 0.72, iterations=1)
    distance = ndimage.distance_transform_edt(mask)

    medial_axis = _medial_axis(distance, 0.9)

    assert int(mask.sum()) == 641
    assert int(medial_axis.sum()) == 629

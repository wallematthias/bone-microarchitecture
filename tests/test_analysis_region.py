"""Regression: a reporting ROI must constrain spheres, not clip their EDT phase."""

import numpy as np
import pytest

from bone_microarchitecture import compute_microarchitecture
from bone_microarchitecture.thickness import hildebrand_thickness_map


def test_analysis_region_reassigns_spheres_without_creating_a_phase_wall():
    z, y, x = np.indices((17, 17, 17))
    cavity = (z - 8) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 <= 25
    peri = np.ones(cavity.shape, dtype=bool)
    region = x >= 10
    kwargs = dict(
        bone_mask=~cavity,
        periosteal_mask=peri,
        trabecular_mask=peri,
        spacing=(1, 1, 1),
        thickness_backend="cpu",
    )
    native = compute_microarchitecture(**kwargs)
    restricted = compute_microarchitecture(**kwargs, analysis_mask=region)
    expected = hildebrand_thickness_map(
        cavity, (1, 1, 1), backend="cpu", domain_mask=region
    )
    np.testing.assert_array_equal(restricted.maps["Tb.Sp"], expected)
    assert restricted.maps["Tb.Sp"][8, 8, 11] < native.maps["Tb.Sp"][8, 8, 11]
    clipped = hildebrand_thickness_map(cavity & region, (1, 1, 1), backend="cpu")
    assert not np.array_equal(expected, clipped)
    assert restricted.measurements["Tb.TV"] == 17 * 17 * 7
    assert all(not np.any(array[~region]) for array in restricted.maps.values())


def test_all_region_preserves_native_measurements_and_maps():
    peri = np.ones((7, 7, 7), dtype=bool)
    bone = np.zeros_like(peri)
    bone[2:5, 2:5, 2:5] = True
    kwargs = dict(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=peri,
        spacing=(0.1, 0.1, 0.1),
        thickness_backend="cpu",
    )
    native = compute_microarchitecture(**kwargs)
    restricted = compute_microarchitecture(**kwargs, analysis_mask=peri)
    assert native.measurements == restricted.measurements
    for name in native.maps:
        np.testing.assert_array_equal(native.maps[name], restricted.maps[name])


def test_region_does_not_reselect_or_delete_a_native_cortical_pore():
    peri = np.zeros((11, 11, 11), dtype=bool)
    peri[:, 1:10, 1:10] = True
    trab = np.zeros_like(peri)
    bone = peri.copy()
    bone[2:9, 5, 5] = False
    region = np.zeros_like(peri)
    region[5, 5, 5] = True
    result = compute_microarchitecture(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab,
        cortical_mask=peri,
        analysis_mask=region,
        spacing=(1, 1, 1),
        thickness_backend="cpu",
    )
    assert result.measurements["Ct.Po.V"] == 1
    assert result.maps["Ct.Po.Mask"][5, 5, 5] == 1


def test_analysis_region_rejects_misaligned_shape():
    with pytest.raises(ValueError, match="shape"):
        compute_microarchitecture(
            periosteal_mask=np.ones((3, 3, 3)),
            trabecular_mask=np.ones((3, 3, 3)),
            analysis_mask=np.ones((2, 3, 3)),
            spacing=(1, 1, 1),
            thickness_backend="cpu",
        )

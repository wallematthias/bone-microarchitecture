from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from bone_microarchitecture.thickness import (
    _medial_axis,
    hildebrand_thickness_map,
    local_thickness_map,
    separation_map,
    summary,
    trabecular_number_map,
)


def test_local_thickness_returns_zero_map_for_empty_mask():
    result = local_thickness_map(np.zeros((3, 3, 3), dtype=bool), (1.0, 1.0, 1.0))

    assert result.dtype == np.float32
    assert result.shape == (3, 3, 3)
    assert np.count_nonzero(result) == 0


def test_local_thickness_respects_anisotropic_spacing():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    result = local_thickness_map(mask, (0.5, 1.0, 2.0))

    assert result[1, 1, 1] == pytest.approx(1.0)


def test_hildebrand_thickness_rejects_unknown_backend():
    mask = np.ones((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="Thickness backend"):
        hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="cuda")


def test_hildebrand_thickness_restricts_output_to_input_mask():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    result = hildebrand_thickness_map(mask, (1.0, 1.0, 1.0), backend="cpu")

    assert result.dtype == np.float32
    assert np.count_nonzero(result[~mask]) == 0
    assert result[2, 2, 2] > 0


def test_medial_axis_uses_dominated_center_pruning_only():
    rng = np.random.default_rng(2)
    mask = ndimage.binary_closing(rng.random((12, 12, 12)) > 0.72, iterations=1)
    distance = ndimage.distance_transform_edt(mask)

    medial_axis = _medial_axis(distance, 0.9)

    assert int(mask.sum()) == 641
    assert int(medial_axis.sum()) == 629


def test_separation_map_measures_space_inside_periosteal_mask():
    peri = np.ones((3, 3, 3), dtype=bool)
    trab = np.zeros_like(peri)
    trab[1, 1, 1] = True

    result = separation_map(peri, trab, (1.0, 1.0, 1.0))

    assert result[1, 1, 1] == 0.0
    assert np.count_nonzero(result) == 26


def test_trabecular_number_map_is_zero_outside_domain():
    domain = np.zeros((5, 5, 5), dtype=bool)
    domain[1:4, 1:4, 1:4] = True
    bone = np.zeros_like(domain)
    bone[2, 2, 2] = True

    result = trabecular_number_map(bone, domain, (1.0, 1.0, 1.0), backend="cpu")

    assert result.dtype == np.float32
    assert np.count_nonzero(result[~domain]) == 0
    assert np.count_nonzero(result[domain]) > 0


def test_summary_ignores_zero_negative_and_nonfinite_values():
    values = np.array([0.0, -1.0, 1.0, 3.0, np.nan, np.inf])

    result = summary(values)

    assert result["mean"] == pytest.approx(2.0)
    assert result["median"] == pytest.approx(2.0)
    assert result["sd"] == pytest.approx(1.0)
    assert result["min"] == 1.0
    assert result["max"] == 3.0

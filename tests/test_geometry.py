from __future__ import annotations

import numpy as np
import pytest

from bone_microarchitecture.geometry import (
    as_bool_mask,
    validate_same_shape,
    validate_spacing,
    voxel_volume,
)


def test_as_bool_mask_accepts_3d_numeric_arrays():
    mask = as_bool_mask(np.array([[[0, 2], [-1, 0]]]), "mask")

    assert mask.dtype == bool
    assert mask.tolist() == [[[False, True], [False, False]]]


def test_as_bool_mask_rejects_non_3d_arrays():
    with pytest.raises(ValueError, match="3D mask"):
        as_bool_mask(np.ones((3, 3)), "trabecular_mask")


def test_validate_spacing_returns_three_positive_floats():
    assert validate_spacing([0.061, 0.061, 0.061]) == (0.061, 0.061, 0.061)


@pytest.mark.parametrize("spacing", [(0.1, 0.1), (0.1, 0.0, 0.1), (0.1, -0.1, 0.1)])
def test_validate_spacing_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="three positive"):
        validate_spacing(spacing)


def test_validate_same_shape_returns_shared_shape():
    arrays = {
        "periosteal_mask": np.zeros((2, 3, 4)),
        "trabecular_mask": np.ones((2, 3, 4)),
        "cortical_mask": None,
    }

    assert validate_same_shape(arrays) == (2, 3, 4)


def test_validate_same_shape_reports_each_mismatched_input():
    arrays = {
        "periosteal_mask": np.zeros((2, 3, 4)),
        "trabecular_mask": np.ones((2, 3, 5)),
    }

    with pytest.raises(ValueError, match=r"periosteal_mask=\(2, 3, 4\).*trabecular_mask=\(2, 3, 5\)"):
        validate_same_shape(arrays)


def test_voxel_volume_uses_array_ordered_spacing():
    assert voxel_volume((0.1, 0.2, 0.3)) == pytest.approx(0.006)

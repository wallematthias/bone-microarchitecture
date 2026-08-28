from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def as_bool_mask(mask, name: str) -> np.ndarray:
    """Validate and convert an input array to a 3D boolean mask."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D mask.")
    return array > 0


def validate_spacing(spacing) -> tuple[float, float, float]:
    """Return spacing as three positive floats."""
    values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("spacing must contain three positive values.")
    return values


def validate_same_shape(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    """Validate that all supplied arrays share one shape."""
    shapes = {name: tuple(array.shape) for name, array in arrays.items() if array is not None}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        detail = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise ValueError(f"All masks and images must have the same shape: {detail}.")
    if not unique_shapes:
        raise ValueError("At least one mask is required.")
    return next(iter(unique_shapes))


def voxel_volume(spacing: tuple[float, float, float]) -> float:
    """Return voxel volume in mm^3 for array-ordered spacing."""
    return float(np.prod(np.asarray(spacing, dtype=float)))

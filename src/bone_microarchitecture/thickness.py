from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage


_NEIGHBOR_OFFSETS = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
)


def local_thickness_map(mask, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return a bounded distance-transform thickness preview.

    This fast map assigns each foreground voxel twice its distance to the
    background. It is useful as a lightweight preview or fallback, but it is not
    the maximal-sphere local thickness used for primary reporting.

    Args:
        mask: 3D binary object mask.
        spacing: Voxel spacing in millimetres, ordered like the array axes.

    Returns:
        Float32 thickness map in millimetres, zero outside ``mask``.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    distance = ndimage.distance_transform_edt(binary, sampling=spacing)
    distance *= 2.0
    return distance.astype(np.float32, copy=False)


def hildebrand_thickness_map(
    mask,
    spacing: tuple[float, float, float],
    *,
    backend: str = "auto",
    center_dominance_mm: float | None = None,
    diameter_margin_voxels: float = 0.5,
) -> np.ndarray:
    """Return a Hildebrand-style maximal-sphere local thickness map.

    This follows the local-thickness concept used in ORMiR-XCT: compute the
    Euclidean distance transform, keep non-dominated sphere centers, and assign
    each object voxel the largest sphere diameter that contains it. The
    implementation here is independent and keeps the candidate-selection rule
    deliberately small: a center is skipped only when a neighboring center has a
    clearly larger inscribed radius.

    Args:
        mask: 3D binary object mask.
        spacing: Voxel spacing in millimetres, ordered like the array axes.
        backend: Diameter-accumulation backend: ``"auto"``, ``"cpu"``, ``"mps"``, or
            ``"opencl"``.
        center_dominance_mm: Minimum radius advantage, in millimetres, required for a
            neighboring center to dominate the current center. Defaults to
            ``0.9 * min(spacing)``.
        diameter_margin_voxels: Small voxel-scaled subtraction from sphere radius
            before reporting diameter. This keeps assignment conservative at the
            discretized boundary.

    Returns:
        Float32 local-thickness map in millimetres, zero outside ``mask``.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    spacing = tuple(float(value) for value in spacing)
    distance = ndimage.distance_transform_edt(binary, sampling=spacing)
    dominance_margin = float(center_dominance_mm if center_dominance_mm is not None else min(spacing) * 0.9)
    medial_axis = _medial_axis(distance, dominance_margin)
    backend = str(backend or "auto").strip().lower()
    if backend == "auto":
        backend = default_thickness_backend()
    if backend == "mps":
        return _mask_output(_accumulate_local_diameters_mps(distance, medial_axis, spacing, diameter_margin_voxels), binary)
    if backend == "opencl":
        return _mask_output(_accumulate_local_diameters_opencl(distance, medial_axis, spacing, diameter_margin_voxels), binary)
    if backend == "cpu":
        return _mask_output(_accumulate_local_diameters_cpu(distance, medial_axis, spacing, diameter_margin_voxels), binary)
    raise ValueError("Thickness backend must be one of: auto, cpu, mps, or opencl.")


def default_thickness_backend() -> str:
    """Choose the fastest available diameter-accumulation backend for this machine."""
    if sys.platform == "darwin":
        try:
            from .metal import is_metal_available

            if is_metal_available():
                return "mps"
        except Exception:
            pass
    else:
        try:
            from .opencl import is_opencl_available

            if is_opencl_available():
                return "opencl"
        except Exception:
            pass
    return "cpu"


def separation_map(periosteal_mask, trabecular_mask, spacing: tuple[float, float, float]) -> np.ndarray:
    """Return a marrow-space thickness map inside the periosteal compartment.

    Args:
        periosteal_mask: Full periosteal compartment mask.
        trabecular_mask: Trabecular bone mask.
        spacing: Voxel spacing in millimetres, ordered like the array axes.

    Returns:
        Fast distance-transform separation map in millimetres.
    """
    peri = np.asarray(periosteal_mask) > 0
    trab = np.asarray(trabecular_mask) > 0
    marrow = peri & ~trab
    return local_thickness_map(marrow, spacing)


def trabecular_number_map(
    bone_mask,
    domain_mask,
    spacing: tuple[float, float, float],
    *,
    thickness_method: str = "hildebrand",
    backend: str = "auto",
    material_center_dominance_voxels: float = 0.5,
    spacing_center_dominance_voxels: float = 0.9,
    diameter_margin_voxels: float = 0.5,
) -> np.ndarray:
    """Estimate local trabecular number from ridge-to-ridge spacing.

    The map is built in two stages. First, trabecular bone ridges are extracted
    from the bone distance transform. Second, a local spacing field is estimated
    in the trabecular domain after removing those ridges. Local trabecular number
    is the inverse of that spacing field.

    Args:
        bone_mask: Trabecular bone phase mask.
        domain_mask: Trabecular compartment mask.
        spacing: Voxel spacing in millimetres, ordered like the array axes.
        thickness_method: Spacing-field thickness method. ``"hildebrand"`` uses
            maximal-sphere spacing; ``"edt"`` uses the faster distance-transform
            approximation.
        backend: Diameter-accumulation backend used for the local spacing field.
        material_center_dominance_voxels: Voxel-scaled dominance threshold for extracting
            trabecular material ridges.
        spacing_center_dominance_voxels: Voxel-scaled dominance threshold for
            extracting spacing-field centers.
        diameter_margin_voxels: Voxel-scaled conservative radius adjustment for
            sphere assignment.

    Returns:
        Float32 trabecular number map in 1/mm, zero outside valid domain voxels.
    """
    bone = np.asarray(bone_mask) > 0
    domain = np.asarray(domain_mask) > 0
    if not bone.any() or not domain.any():
        return np.zeros(domain.shape, dtype=np.float32)
    spacing = tuple(float(value) for value in spacing)
    bone = bone & domain
    bone_distance = ndimage.distance_transform_edt(bone, sampling=spacing)
    material_ridge = _medial_axis(bone_distance, float(material_center_dominance_voxels) * min(spacing))
    inter_axis = domain & ~material_ridge
    method = str(thickness_method or "hildebrand").strip().lower()
    if method == "edt":
        spacing_map = local_thickness_map(inter_axis, spacing)
    elif method == "hildebrand":
        spacing_map = hildebrand_thickness_map(
            inter_axis,
            spacing,
            backend=backend,
            center_dominance_mm=float(spacing_center_dominance_voxels) * min(spacing),
            diameter_margin_voxels=diameter_margin_voxels,
        )
    else:
        raise ValueError("Trabecular number thickness_method must be 'hildebrand' or 'edt'.")
    number = np.zeros(domain.shape, dtype=np.float32)
    valid = domain & np.isfinite(spacing_map) & (spacing_map > 0)
    number[valid] = (1.0 / spacing_map[valid]).astype(np.float32, copy=False)
    return number


def summary(values) -> dict[str, float]:
    """Summarize positive finite values with mean, percentiles, and range."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "sd": 0.0,
            "p5": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=0)),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _medial_axis(distance: np.ndarray, dominance_margin: float) -> np.ndarray:
    padded = np.pad(distance, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1, 1:-1]
    inside = center > 0
    dominated = _dominated_center_mask(padded, center, dominance_margin)
    return inside & ~dominated


def _dominated_center_mask(padded: np.ndarray, center: np.ndarray, dominance_margin: float) -> np.ndarray:
    dominated = np.zeros(center.shape, dtype=bool)
    for dz, dy, dx in _NEIGHBOR_OFFSETS:
        neighbor = padded[
            1 + dz : 1 + dz + center.shape[0],
            1 + dy : 1 + dy + center.shape[1],
            1 + dx : 1 + dx + center.shape[2],
        ]
        dominated |= (center + dominance_margin) <= neighbor
    return dominated


def _mask_output(thickness: np.ndarray, binary: np.ndarray) -> np.ndarray:
    thickness = np.asarray(thickness, dtype=np.float32)
    thickness[~binary] = 0
    return thickness


def _sphere_centers(distance: np.ndarray, medial_axis: np.ndarray):
    z, y, x = np.nonzero(medial_axis)
    radius = distance[medial_axis].astype(np.float32, copy=False)
    if radius.size == 0:
        return z, y, x, radius
    order = np.argsort(radius)[::-1]
    return z[order], y[order], x[order], radius[order]


def _accumulate_local_diameters_cpu(
    distance: np.ndarray,
    medial_axis: np.ndarray,
    spacing: tuple[float, float, float],
    diameter_margin_voxels: float,
) -> np.ndarray:
    shape = distance.shape
    thickness = np.zeros(shape, dtype=np.float32)
    seed_z, seed_y, seed_x, seed_radius = _sphere_centers(distance, medial_axis)
    diameter_margin = float(diameter_margin_voxels) * min(spacing)
    inclusion_tolerance = _sphere_inclusion_tolerance(spacing)
    for zc, yc, xc, radius in zip(seed_z, seed_y, seed_x, seed_radius):
        value = np.float32(max(2.0 * (float(radius) - diameter_margin), 0.0))
        if value <= 0:
            continue
        z0, z1, y0, y1, x0, x1 = _sphere_bounds(shape, int(zc), int(yc), int(xc), float(radius), spacing)
        zz, yy, xx = _physical_offsets(z0, z1, y0, y1, x0, x1, int(zc), int(yc), int(xc), spacing)
        in_sphere = (zz * zz + yy * yy + xx * xx) <= (float(radius) * float(radius) + inclusion_tolerance)
        sub = thickness[z0:z1, y0:y1, x0:x1]
        np.maximum(sub, np.where(in_sphere, value, 0).astype(np.float32, copy=False), out=sub)
    return thickness


def _accumulate_local_diameters_mps(
    distance: np.ndarray,
    medial_axis: np.ndarray,
    spacing: tuple[float, float, float],
    diameter_margin_voxels: float,
) -> np.ndarray:
    from .metal import metal_hildebrand_thickness_map

    seed_z, seed_y, seed_x, seed_radius = _sphere_centers(distance, medial_axis)
    return metal_hildebrand_thickness_map(
        shape=distance.shape,
        seed_z=seed_z,
        seed_y=seed_y,
        seed_x=seed_x,
        seed_radius=seed_radius,
        spacing=spacing,
        diameter_margin=float(diameter_margin_voxels) * min(spacing),
        inclusion_tolerance=_sphere_inclusion_tolerance(spacing),
    )


def _accumulate_local_diameters_opencl(
    distance: np.ndarray,
    medial_axis: np.ndarray,
    spacing: tuple[float, float, float],
    diameter_margin_voxels: float,
) -> np.ndarray:
    from .opencl import opencl_hildebrand_thickness_map

    seed_z, seed_y, seed_x, seed_radius = _sphere_centers(distance, medial_axis)
    return opencl_hildebrand_thickness_map(
        shape=distance.shape,
        seed_z=seed_z,
        seed_y=seed_y,
        seed_x=seed_x,
        seed_radius=seed_radius,
        spacing=spacing,
        diameter_margin=float(diameter_margin_voxels) * min(spacing),
        inclusion_tolerance=_sphere_inclusion_tolerance(spacing),
    )


def _sphere_bounds(shape, zc: int, yc: int, xc: int, radius: float, spacing: tuple[float, float, float]):
    z_extent, y_extent, x_extent = (int(np.ceil(radius / value)) for value in spacing)
    z0 = max(0, zc - z_extent)
    z1 = min(shape[0], zc + z_extent + 1)
    y0 = max(0, yc - y_extent)
    y1 = min(shape[1], yc + y_extent + 1)
    x0 = max(0, xc - x_extent)
    x1 = min(shape[2], xc + x_extent + 1)
    return z0, z1, y0, y1, x0, x1


def _physical_offsets(
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    zc: int,
    yc: int,
    xc: int,
    spacing: tuple[float, float, float],
):
    z_offsets = (np.arange(z0, z1, dtype=np.float32) - zc) * spacing[0]
    y_offsets = (np.arange(y0, y1, dtype=np.float32) - yc) * spacing[1]
    x_offsets = (np.arange(x0, x1, dtype=np.float32) - xc) * spacing[2]
    return np.meshgrid(z_offsets, y_offsets, x_offsets, indexing="ij")


def _sphere_inclusion_tolerance(spacing: tuple[float, float, float]) -> float:
    spacing_min = min(float(value) for value in spacing)
    return max(spacing_min * spacing_min, 1.0) * 1.0e-6

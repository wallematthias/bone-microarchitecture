from __future__ import annotations

import numpy as np

from .geometry import voxel_volume


def count_volume(mask, spacing: tuple[float, float, float]) -> float:
    """Return the physical volume represented by non-zero voxels.

    Args:
        mask: 3D binary mask. Non-zero values are counted.
        spacing: Voxel spacing in millimetres, ordered like the array axes.

    Returns:
        Volume in mm^3.
    """
    return float(np.count_nonzero(mask) * voxel_volume(spacing))


def masked_mean_sd(image, mask) -> tuple[float, float]:
    """Return mean and standard deviation inside a mask.

    Non-finite image values are ignored. Empty masks return ``(0.0, 0.0)``.
    """
    values = np.asarray(image, dtype=float)[np.asarray(mask) > 0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0))


def compartment_metrics(
    *,
    bone_mask,
    periosteal_mask,
    trabecular_mask,
    cortical_mask=None,
    spacing: tuple[float, float, float],
    mean_tb_th: float = 0.0,
) -> dict[str, float]:
    """Calculate scalar compartment measures from binary masks.

    The trabecular compartment is the intersection of ``trabecular_mask`` and
    ``periosteal_mask``, excluding the cortical compartment when one is supplied.
    Bone volume measures use ``bone_mask`` intersected with the relevant
    compartment. Ratio outputs are unitless fractions.

    Calculated parameters:
        ``Tb.BV``: trabecular bone volume, in mm^3.
        ``Tb.TV``: trabecular compartment volume, in mm^3.
        ``Tb.BV/TV``: ``Tb.BV / Tb.TV``, as a fraction.
        ``Ct.BV``: cortical bone volume, in mm^3.
        ``Ct.TV``: cortical compartment volume, in mm^3.
        ``Ct.Po.V``: ``Ct.TV - Ct.BV``, in mm^3.
        ``Ct.Po``: ``Ct.Po.V / Ct.TV``, as a fraction.
        ``Tb.N``: fallback scalar ``Tb.BV/TV / mean(Tb.Th)``. The pipeline
        replaces this with the map-based trabecular number estimate.
    """
    trab_region = np.asarray(trabecular_mask) > 0
    peri = np.asarray(periosteal_mask) > 0
    bone = np.asarray(bone_mask) > 0 if bone_mask is not None else trab_region
    cort = np.asarray(cortical_mask) > 0 if cortical_mask is not None else np.zeros_like(trab_region)
    trab_region = trab_region & peri & ~cort
    cort_region = cort
    trab_bone = bone & trab_region
    cortical_bone = bone & cort_region
    tb_bv = count_volume(trab_bone, spacing)
    tb_tv = count_volume(trab_region, spacing)
    ct_bv = count_volume(cortical_bone, spacing)
    ct_tv = count_volume(cort_region, spacing)
    bvtv = tb_bv / tb_tv if tb_tv else 0.0
    ct_po = max(ct_tv - ct_bv, 0.0) / ct_tv if ct_tv else 0.0
    metrics = {
        "Tb.BV/TV": bvtv,
        "Tb.BV": tb_bv,
        "Tb.TV": tb_tv,
        "Ct.BV": ct_bv,
        "Ct.TV": ct_tv,
        "Ct.Po.V": max(ct_tv - ct_bv, 0.0),
        "Ct.Po": ct_po,
    }
    metrics["Tb.N"] = bvtv / mean_tb_th if mean_tb_th else 0.0
    return metrics

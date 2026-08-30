from __future__ import annotations

import numpy as np

from .geometry import as_bool_mask, validate_same_shape, validate_spacing
from .metrics import compartment_metrics, masked_mean_sd
from .results import MicroarchitectureResult
from .thickness import (
    default_thickness_backend,
    hildebrand_thickness_map,
    local_thickness_map,
    summary,
    trabecular_number_map,
)


def compute_microarchitecture(
    *,
    periosteal_mask,
    trabecular_mask,
    spacing,
    bone_mask=None,
    cortical_mask=None,
    grayscale=None,
    thickness_method: str = "hildebrand",
    thickness_backend: str = "auto",
) -> MicroarchitectureResult:
    """Compute HR-pQCT microarchitecture measurements from aligned arrays.

    This function is intentionally image-I/O agnostic. Callers must load images,
    calibrate grayscale data when BMD is desired, and resample all masks/images
    to a shared voxel grid before calling it.

    Masks are interpreted as:
        ``periosteal_mask``: full analysis compartment.
        ``trabecular_mask``: trabecular compartment.
        ``cortical_mask``: optional cortical compartment.
        ``bone_mask``: mineralized bone phase. If omitted, the trabecular mask is
        treated as the bone phase for backward-compatible trabecular-only use.

    Calculated trabecular outputs:
        ``Tb.BMD``: mean grayscale/BMD inside the trabecular compartment.
        ``Tb.BV/TV``: trabecular bone volume divided by trabecular total volume,
        reported as a unitless fraction.
        ``Tb.Th``: mean local thickness of trabecular bone, in mm.
        ``Tb.Sp``: mean local thickness of non-bone space in the trabecular
        compartment, in mm.
        ``Tb.N``: mean inverse ridge-to-ridge spacing estimate, in 1/mm.
        ``Tb.1/N.SD``: standard deviation of ridge-to-ridge spacing, in mm.
        ``Tb.BV`` and ``Tb.TV``: trabecular bone and compartment volumes, in mm^3.

    Calculated cortical outputs when ``cortical_mask`` is provided:
        ``Ct.BMD``: mean grayscale/BMD inside the cortical compartment.
        ``Ct.Th``: mean local thickness of cortical bone, in mm.
        ``Ct.Po``: cortical pore volume divided by cortical total volume,
        reported as a unitless fraction.
        ``Ct.Po.V``: cortical pore volume, in mm^3.
        ``Ct.Po.Dm``: mean local pore diameter, in mm.
        ``Ct.BV`` and ``Ct.TV``: cortical bone and compartment volumes, in mm^3.

    Map-backed parameters include distribution statistics in
    :func:`bone_microarchitecture.results.measurement_rows`.
    """
    spacing = validate_spacing(spacing)
    peri = as_bool_mask(periosteal_mask, "periosteal_mask")
    trab = as_bool_mask(trabecular_mask, "trabecular_mask")
    bone = as_bool_mask(bone_mask, "bone_mask") if bone_mask is not None else trab
    cort = as_bool_mask(cortical_mask, "cortical_mask") if cortical_mask is not None else None
    image = None if grayscale is None else np.asarray(grayscale, dtype=np.float32)
    validate_same_shape(
        {
            "periosteal_mask": peri,
            "trabecular_mask": trab,
            "bone_mask": bone,
            "cortical_mask": cort,
            "grayscale": image,
        }
    )

    thickness_method = str(thickness_method or "hildebrand").strip().lower()
    thickness_backend = str(thickness_backend or "auto").strip().lower()
    resolved_thickness_backend = default_thickness_backend() if thickness_backend == "auto" else thickness_backend
    trab_region = trab & peri
    if cort is not None:
        trab_region = trab_region & ~cort
        cort_region = cort & peri
    else:
        cort_region = None
    trab_bone = bone & trab_region
    cort_bone = None if cort_region is None else bone & cort_region
    tb_th_map = _thickness_map(
        trab_bone,
        spacing,
        thickness_method=thickness_method,
        thickness_backend=resolved_thickness_backend,
    )
    tb_sp_map = _thickness_map(
        trab_region & ~trab_bone,
        spacing,
        thickness_method=thickness_method,
        thickness_backend=resolved_thickness_backend,
    )
    tb_th = summary(tb_th_map[trab_bone])
    tb_sp = summary(tb_sp_map[trab_region & ~trab_bone])
    tb_n_map = trabecular_number_map(
        trab_bone,
        trab_region,
        spacing,
        thickness_method=thickness_method,
        backend=resolved_thickness_backend,
    )
    tb_n = summary(tb_n_map[trab_region])
    tb_inverse_number_map = np.zeros(trab_region.shape, dtype=np.float32)
    valid_tb_n = trab_region & np.isfinite(tb_n_map) & (tb_n_map > 0)
    tb_inverse_number_map[valid_tb_n] = (1.0 / tb_n_map[valid_tb_n]).astype(np.float32, copy=False)
    tb_inverse_number = summary(tb_inverse_number_map[valid_tb_n])

    metrics = compartment_metrics(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab,
        cortical_mask=cort,
        spacing=spacing,
        mean_tb_th=tb_th["mean"],
    )
    metrics["Tb.N"] = tb_n["mean"]
    metrics.update(
        {
            "Tb.Th": tb_th["mean"],
            "Tb.Th SD": tb_th["sd"],
            "Tb.Th Min": tb_th["min"],
            "Tb.Th Max": tb_th["max"],
            "Tb.Sp": tb_sp["mean"],
            "Tb.Sp SD": tb_sp["sd"],
            "Tb.Sp Min": tb_sp["min"],
            "Tb.Sp Max": tb_sp["max"],
            "Tb.N Median": tb_n["median"],
            "Tb.N SD": tb_n["sd"],
            "Tb.N P5": tb_n["p5"],
            "Tb.N P25": tb_n["p25"],
            "Tb.N P75": tb_n["p75"],
            "Tb.N P95": tb_n["p95"],
            "Tb.N Min": tb_n["min"],
            "Tb.N Max": tb_n["max"],
            "Tb.1/N.SD": tb_inverse_number["sd"],
        }
    )

    maps = {"Tb.Th": tb_th_map, "Tb.Sp": tb_sp_map, "Tb.N": tb_n_map}

    if cort_bone is not None:
        ct_th_map = _thickness_map(
            cort_bone,
            spacing,
            thickness_method=thickness_method,
            thickness_backend=resolved_thickness_backend,
        )
        ct_th = summary(ct_th_map[cort_bone])
        metrics.update(
            {
                "Ct.Th": ct_th["mean"],
                "Ct.Th SD": ct_th["sd"],
                "Ct.Th Min": ct_th["min"],
                "Ct.Th Max": ct_th["max"],
            }
        )
        maps["Ct.Th"] = ct_th_map
        pore_map = _thickness_map(
            cort_region & ~cort_bone,
            spacing,
            thickness_method=thickness_method,
            thickness_backend=resolved_thickness_backend,
        )
        pore_summary = summary(pore_map[cort_region & ~cort_bone])
        metrics.update(
            {
                "Ct.Po.Dm": pore_summary["mean"],
                "Ct.Po.Dm SD": pore_summary["sd"],
                "Ct.Po.Dm Min": pore_summary["min"],
                "Ct.Po.Dm Max": pore_summary["max"],
            }
        )
        maps["Ct.Po.Dm"] = pore_map

    if image is not None:
        tb_mean, tb_sd = masked_mean_sd(image, trab_region)
        metrics["Tb.BMD"] = tb_mean
        metrics["Tb.BMD SD"] = tb_sd
        maps["Tb.BMD"] = np.where(trab_region, image, 0).astype(np.float32)
        if cort_region is not None:
            ct_mean, ct_sd = masked_mean_sd(image, cort_region)
            metrics["Ct.BMD"] = ct_mean
            metrics["Ct.BMD SD"] = ct_sd
            maps["Ct.BMD"] = np.where(cort_region, image, 0).astype(np.float32)

    return MicroarchitectureResult(
        measurements=metrics,
        maps=maps,
        metadata={
            "thickness_method": thickness_method,
            "thickness_backend": resolved_thickness_backend,
        },
    )


def _thickness_map(mask, spacing, *, thickness_method: str, thickness_backend: str):
    """Dispatch one binary mask to the requested thickness-map implementation."""
    if thickness_method in {"edt", "distance", "distance_transform"}:
        return local_thickness_map(mask, spacing)
    if thickness_method in {"hildebrand", "sphere_fitting", "sphere-fitting", "exact"}:
        return hildebrand_thickness_map(mask, spacing, backend=thickness_backend)
    raise ValueError("Thickness method must be one of: edt or hildebrand.")

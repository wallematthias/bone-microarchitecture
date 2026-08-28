from __future__ import annotations

import numpy as np
import pytest

from bone_microarchitecture.metrics import compartment_metrics, count_volume, masked_mean_sd


def test_count_volume_counts_nonzero_voxels_in_physical_units():
    mask = np.array([[[0, 1], [2, 0]]])

    assert count_volume(mask, (0.1, 0.2, 0.3)) == pytest.approx(2 * 0.006)


def test_masked_mean_sd_ignores_nonfinite_values():
    image = np.array([[[1.0, np.nan], [3.0, np.inf]]])
    mask = np.ones(image.shape, dtype=bool)

    mean, sd = masked_mean_sd(image, mask)

    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx(1.0)


def test_masked_mean_sd_returns_zero_for_empty_valid_region():
    mean, sd = masked_mean_sd(np.array([[[np.nan]]]), np.ones((1, 1, 1), dtype=bool))

    assert mean == 0.0
    assert sd == 0.0


def test_compartment_metrics_report_volumes_and_ratios_as_fractions():
    peri = np.ones((2, 2, 2), dtype=bool)
    trab = np.zeros_like(peri)
    trab[:, :, 0] = True
    cort = peri & ~trab
    bone = np.zeros_like(peri)
    bone[0, 0, 0] = True
    bone[0, 0, 1] = True

    metrics = compartment_metrics(
        bone_mask=bone,
        periosteal_mask=peri,
        trabecular_mask=trab,
        cortical_mask=cort,
        spacing=(0.5, 0.5, 0.5),
        mean_tb_th=0.25,
    )

    assert metrics["Tb.BV"] == pytest.approx(0.125)
    assert metrics["Tb.TV"] == pytest.approx(0.5)
    assert metrics["Tb.BV/TV"] == pytest.approx(0.25)
    assert metrics["Ct.BV"] == pytest.approx(0.125)
    assert metrics["Ct.TV"] == pytest.approx(0.5)
    assert metrics["Ct.Po.V"] == pytest.approx(0.375)
    assert metrics["Ct.Po"] == pytest.approx(0.75)
    assert metrics["Tb.N"] == pytest.approx(1.0)


def test_compartment_metrics_handles_missing_cortex():
    mask = np.ones((2, 2, 2), dtype=bool)

    metrics = compartment_metrics(
        bone_mask=mask,
        periosteal_mask=mask,
        trabecular_mask=mask,
        spacing=(1.0, 1.0, 1.0),
    )

    assert metrics["Tb.BV"] == pytest.approx(8.0)
    assert metrics["Tb.TV"] == pytest.approx(8.0)
    assert metrics["Ct.BV"] == 0.0
    assert metrics["Ct.TV"] == 0.0
    assert metrics["Ct.Po"] == 0.0

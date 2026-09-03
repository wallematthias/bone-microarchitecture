from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path

import numpy as np


MEASUREMENT_ORDER = (
    "Tt.BMD",
    "Tb.BMD",
    "Tb.BV/TV",
    "Tb.Th",
    "Tb.Sp",
    "Tb.N",
    "Tb.1/N.SD",
    "Tb.BV",
    "Tb.TV",
    "Ct.BMD",
    "Ct.Th",
    "Ct.Po",
    "Ct.Po.V",
    "Ct.Po.Dm",
    "Ct.BV",
    "Ct.TV",
)

SUMMARY_COLUMNS = (
    "Parameter",
    "Mean",
    "Median",
    "SD",
    "P5",
    "P25",
    "P75",
    "P95",
    "Min",
    "Max",
    "Units",
)

_DISTRIBUTION_STAT_KEYS = {
    "Tb.Th": ("Tb.Th SD", "Tb.Th Min", "Tb.Th Max"),
    "Tb.Sp": ("Tb.Sp SD", "Tb.Sp Min", "Tb.Sp Max"),
    "Tb.N": (
        "Tb.N Median",
        "Tb.N SD",
        "Tb.N P5",
        "Tb.N P25",
        "Tb.N P75",
        "Tb.N P95",
        "Tb.N Min",
        "Tb.N Max",
    ),
    "Ct.Th": ("Ct.Th SD", "Ct.Th Min", "Ct.Th Max"),
    "Ct.Po.Dm": ("Ct.Po.Dm SD", "Ct.Po.Dm Min", "Ct.Po.Dm Max"),
    "Tb.BMD": ("Tb.BMD SD",),
    "Tt.BMD": ("Tt.BMD SD",),
    "Ct.BMD": ("Ct.BMD SD",),
}
_SECONDARY_MEASUREMENTS = {name for names in _DISTRIBUTION_STAT_KEYS.values() for name in names}

PARAMETER_DEFINITIONS = {
    "Tt.BMD": "Mean grayscale/BMD value inside the full/periosteal compartment.",
    "Tb.BMD": "Mean grayscale/BMD value inside the trabecular compartment.",
    "Tb.BV/TV": "Trabecular bone volume divided by trabecular total volume, reported as a fraction.",
    "Tb.Th": "Mean maximal-sphere local thickness of trabecular bone.",
    "Tb.Sp": "Mean maximal-sphere local thickness of non-bone space in the trabecular compartment.",
    "Tb.N": "Mean inverse ridge-to-ridge spacing estimate in the trabecular compartment.",
    "Tb.1/N.SD": "Standard deviation of ridge-to-ridge spacing used to estimate trabecular number.",
    "Tb.BV": "Trabecular bone volume.",
    "Tb.TV": "Trabecular compartment volume.",
    "Ct.BMD": "Mean grayscale/BMD value inside the cortical compartment.",
    "Ct.Th": "Mean maximal-sphere local thickness of cortical bone.",
    "Ct.Po": "Cortical pore volume divided by cortical total volume, reported as a fraction.",
    "Ct.Po.V": "Cortical pore volume.",
    "Ct.Po.Dm": "Mean maximal-sphere local diameter of cortical pore space.",
    "Ct.BV": "Cortical bone volume.",
    "Ct.TV": "Cortical compartment volume.",
}


@dataclass(frozen=True)
class MicroarchitectureResult:
    """Container returned by :func:`compute_microarchitecture`.

    Attributes:
        measurements: Scalar parameter values and secondary distribution values.
        maps: Parameter maps keyed by names such as ``"Tb.Th"`` and
            ``"Ct.Po.Dm"``. Non-map scalar parameters are absent.
        metadata: Calculation metadata, for example selected thickness backend.
    """

    measurements: dict[str, float]
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def units_for_measurement(name: str) -> str:
    """Return display units for a measurement name."""
    if name.endswith("BV/TV") or name.endswith(".Po"):
        return "fraction"
    if name.endswith(".BMD") or name.endswith("BMD SD"):
        return "mgHA/cm^3"
    if name.endswith(".BV") or name.endswith(".TV") or name.endswith(".V"):
        return "mm^3"
    if name == "Tb.N":
        return "1/mm"
    if ".Th" in name or ".Sp" in name or name.endswith(".Dm") or name == "Tb.1/N.SD":
        return "mm"
    return ""


def measurement_rows(metrics: dict[str, float], maps: dict[str, np.ndarray] | None = None) -> list[dict[str, object]]:
    """Format measurements as table rows for Slicer display or CSV export.

    Scalar parameters place their value in the ``Mean`` column. Parameters with
    an available map are summarized across positive finite map voxels and include
    median, standard deviation, percentiles, and range.
    """
    ordered = [name for name in MEASUREMENT_ORDER if name in metrics]
    ordered.extend(sorted(name for name in metrics if name not in set(ordered)))
    return [
        _measurement_row(name, metrics, maps or {})
        for name in ordered
        if name not in _SECONDARY_MEASUREMENTS
    ]


def write_measurement_csv(path, metrics: dict[str, float], maps: dict[str, np.ndarray] | None = None) -> None:
    """Write formatted measurement rows to a CSV file."""
    rows = measurement_rows(metrics, maps)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _measurement_row(name: str, metrics: dict[str, float], maps: dict[str, np.ndarray]) -> dict[str, object]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row["Parameter"] = name
    row["Mean"] = float(metrics[name])
    row["Units"] = units_for_measurement(name)
    if name in maps:
        values = _map_values(maps[name])
        if values.size:
            row.update(
                {
                    "Mean": float(values.mean()),
                    "Median": float(np.median(values)),
                    "SD": float(values.std(ddof=0)),
                    "P5": float(np.percentile(values, 5)),
                    "P25": float(np.percentile(values, 25)),
                    "P75": float(np.percentile(values, 75)),
                    "P95": float(np.percentile(values, 95)),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                }
            )
    return row


def _map_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    return values

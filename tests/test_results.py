from __future__ import annotations

import csv

import numpy as np

from bone_microarchitecture.results import (
    MEASUREMENT_ORDER,
    PARAMETER_DEFINITIONS,
    SUMMARY_COLUMNS,
    measurement_rows,
    units_for_measurement,
    write_measurement_csv,
)


def test_all_primary_measurements_have_definitions():
    assert set(MEASUREMENT_ORDER) <= set(PARAMETER_DEFINITIONS)


def test_units_for_measurement_cover_reported_parameter_families():
    assert units_for_measurement("Tb.BV/TV") == "fraction"
    assert units_for_measurement("Ct.Po") == "fraction"
    assert units_for_measurement("Tb.BMD") == "mgHA/cm^3"
    assert units_for_measurement("Tb.BMD SD") == "mgHA/cm^3"
    assert units_for_measurement("Tb.BV") == "mm^3"
    assert units_for_measurement("Ct.Po.V") == "mm^3"
    assert units_for_measurement("Tb.N") == "1/mm"
    assert units_for_measurement("Tb.Th") == "mm"
    assert units_for_measurement("Ct.Po.Dm") == "mm"
    assert units_for_measurement("unknown") == ""


def test_measurement_rows_preserve_order_and_hide_secondary_stat_keys():
    metrics = {
        "Ct.Po.Dm SD": 0.1,
        "Tb.BV/TV": 0.2,
        "Tb.Th": 0.3,
        "Tb.Th SD": 0.4,
        "Extra": 42.0,
    }

    rows = measurement_rows(metrics)

    assert [row["Parameter"] for row in rows] == ["Tb.BV/TV", "Tb.Th", "Extra"]
    assert rows[0]["Mean"] == 0.2
    assert rows[0]["Median"] == ""


def test_measurement_rows_summarize_positive_finite_map_values():
    metrics = {"Tb.Th": 99.0}
    maps = {"Tb.Th": np.array([[[0.0, 1.0], [2.0, np.nan]]], dtype=np.float32)}

    row = measurement_rows(metrics, maps)[0]

    assert row["Parameter"] == "Tb.Th"
    assert row["Mean"] == 1.5
    assert row["Median"] == 1.5
    assert row["SD"] == 0.5
    assert row["Min"] == 1.0
    assert row["Max"] == 2.0


def test_write_measurement_csv_uses_summary_columns(tmp_path):
    path = tmp_path / "measurements.csv"

    write_measurement_csv(path, {"Tb.BV/TV": 0.25})

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(SUMMARY_COLUMNS)
    assert rows[0]["Parameter"] == "Tb.BV/TV"
    assert rows[0]["Mean"] == "0.25"
    assert rows[0]["Units"] == "fraction"

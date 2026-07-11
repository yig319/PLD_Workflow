from pathlib import Path

import pytest

from pld_workflow.analysis import build_parameter_trend, list_available_parameters

pytest.importorskip("pandas")


def test_list_available_parameters_finds_target_parameters():
    sample_root = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data" / "sample_record.json"

    summary = list_available_parameters([sample_root])

    assert not summary.empty
    assert ((summary["section"] == "target_1") & (summary["parameter"] == "Laser Energy (mJ)")).any()


def test_build_parameter_trend_returns_numeric_values():
    sample_root = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data" / "sample_record.json"

    trend = build_parameter_trend([sample_root], parameter="Laser Energy (mJ)", section="target_1")

    assert len(trend) == 1
    assert float(trend.iloc[0]["numeric_value"]) == 245.0
    assert trend.iloc[0]["growth_id"] == "PLD-2026-001"

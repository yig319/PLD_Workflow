from __future__ import annotations

import json
from datetime import datetime

import pytest

from pld_workflow.pulse_tracking import calculate_run_pulses, find_target_pulse_history


def _write_record(path, *, growth_id, timestamp, target, chamber="PLD-1"):
    data = {
        "header": {
            "Growth ID": growth_id,
            "Date": timestamp.strftime("%m/%d/%Y"),
            "time": timestamp.strftime("%H:%M:%S"),
            "Chamber": chamber,
        },
        "target_1": target,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_calculate_run_pulses_separates_target_and_laser_usage():
    assert calculate_run_pulses(100, 1500, 25, 80) == (1625, 1705)


def test_history_uses_saved_checkpoint_then_adds_legacy_run(tmp_path):
    _write_record(
        tmp_path / "first.json",
        growth_id="G-1",
        timestamp=datetime(2026, 8, 20, 10),
        target={
            "Target ID": "SRO-01",
            "Target Material": "SrRuO3",
            "Target Pulses After Run (count)": 10_000,
            "Ablation Pulses (count)": 500,
        },
    )
    _write_record(
        tmp_path / "second.json",
        growth_id="G-2",
        timestamp=datetime(2026, 8, 21, 10),
        target={
            "Target ID": "SRO-01",
            "Target Material": "SrRuO3",
            "Pre-Ablation Pulses (count)": 100,
            "Ablation Pulses (count)": 900,
        },
    )

    result = find_target_pulse_history(
        [tmp_path],
        before_timestamp=datetime(2026, 8, 22, 10),
        target_id="SRO-01",
        material="SrRuO3",
        chamber="PLD-1",
    )

    assert result.before_pulses == 11_000
    assert result.source_growth_id == "G-2"
    assert result.matching_records == 2
    assert not result.used_material_fallback


def test_history_falls_back_to_material_for_legacy_record(tmp_path):
    _write_record(
        tmp_path / "legacy.json",
        growth_id="G-1",
        timestamp=datetime(2026, 8, 20, 10),
        target={
            "Target Material": "SrRuO3",
            "Pre-Ablation Pulses (count)": 100,
            "Ablation Pulses (count)": 900,
        },
    )

    result = find_target_pulse_history(
        [tmp_path],
        before_timestamp=datetime(2026, 8, 22, 10),
        target_id="SRO-01",
        material="srruo3",
        chamber="PLD-1",
    )

    assert result.before_pulses == 1_000
    assert result.used_material_fallback
    assert "verify" in result.warning.lower()


def test_history_excludes_records_at_or_after_new_timestamp(tmp_path):
    _write_record(
        tmp_path / "future.json",
        growth_id="G-future",
        timestamp=datetime(2026, 8, 25, 12),
        target={"Target ID": "SRO-01", "Ablation Pulses (count)": 5000},
    )

    result = find_target_pulse_history(
        [tmp_path],
        before_timestamp=datetime(2026, 8, 25, 11),
        target_id="SRO-01",
        chamber="PLD-1",
    )

    assert result.before_pulses == 0
    assert result.matching_records == 0


def test_material_fallback_rejects_ambiguous_targets_in_one_record(tmp_path):
    data = {
        "header": {"Growth ID": "G-1", "Date": "08/20/2026", "time": "10:00:00", "Chamber": "PLD-1"},
        "target_1": {"Target Material": "SRO", "Ablation Pulses (count)": 100},
        "target_2": {"Target Material": "SRO", "Ablation Pulses (count)": 200},
    }
    (tmp_path / "ambiguous.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unique Target IDs"):
        find_target_pulse_history(
            [tmp_path],
            before_timestamp=datetime(2026, 8, 22, 10),
            material="SRO",
            chamber="PLD-1",
        )

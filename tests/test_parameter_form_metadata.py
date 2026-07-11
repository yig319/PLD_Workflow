from __future__ import annotations

from datetime import datetime

from pld_workflow.form import resolve_header_timestamp
from pld_workflow.parameter_export import write_html_report


def test_resolve_header_timestamp_defaults_to_now_for_templates():
    now = datetime(2026, 4, 24, 15, 16, 17)

    assert resolve_header_timestamp({}, now=now) == ("04/24/2026", "15:16:17")


def test_resolve_header_timestamp_preserves_loaded_values():
    now = datetime(2026, 4, 24, 15, 16, 17)
    header = {"Date": "04/01/2026", "Time": "08:09:10"}

    assert resolve_header_timestamp(header, now=now) == ("04/01/2026", "08:09:10")


def test_html_report_uses_refactored_target_sections(tmp_path):
    report_path = tmp_path / "pld_report.html"
    data = {
        "header": {"Growth ID": "PLD-001", "User Name": "Yichen", "Notes": "Test note"},
        "target_1": {
            "Target Material": "SRO",
            "Offset X (mm)": 0.5,
            "Heater Position X (mm)": 1.0,
            "Mask Width (mm)": 15.0,
            "Laser Voltage (kV)": 24.0,
            "Ablation Temperature (\N{DEGREE SIGN}C)": 650.0,
            "Tilt (deg)": 0.5,
            "Azimuth (deg)": -12.0,
            "Pre-Annealing Temperature (\N{DEGREE SIGN}C)": 800.0,
            "Growth Condition Heating Rate (\N{DEGREE SIGN}C/min)": 20.0,
            "Pre-Ablation Frequency (Hz)": 5.0,
            "Pre-Ablation Pulses (count)": 100.0,
            "Post-Annealing Cooling Rate (\N{DEGREE SIGN}C/min)": 10.0,
            "Ablation Frequency (Hz)": 2.0,
            "Ablation Pulses (count)": 1500.0,
        },
    }

    write_html_report(str(report_path), data)
    html = report_path.read_text(encoding="utf-8")

    # Top-level cards matching the form UI layout
    assert "<h2>Header</h2>" in html
    assert "<h2>Instrument</h2>" in html
    assert "<h2>Preparation</h2>" in html
    assert "<h2>Deposition - SRO</h2>" in html
    assert "<h2>Notes</h2>" in html

    # Instrument sub-sections
    assert "Target Setup" in html
    assert "Heater Position" in html
    assert "Mask and Spot" in html
    assert "Laser" in html
    assert "RHEED Adjustment" in html

    # Preparation sub-sections
    assert "Pre-Annealing" in html
    assert "Growth Condition" in html
    assert "Cool Down" in html

    # Deposition sub-sections
    assert "Pre-Ablation" in html
    assert "Ablation" in html

    # Cool Down only has Cooling Rate (no Temperature or Hold Time)
    assert "Post-Annealing Temperature" not in html
    assert "Post-Annealing Hold Time" not in html

    # Verify value appears in Growth Condition sub-section
    assert "650.0" in html

from __future__ import annotations

from datetime import datetime

from PyQt5.QtWidgets import QApplication

from pld_workflow.form import resolve_header_timestamp
from pld_workflow.parameter_export import coerce_numeric_values, write_html_report


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
            "Additional On-Target Pulses (count)": 25,
            "Off-Target Pulses (count)": 75,
            "Target Pulses Before Run (count)": 10000,
            "Target Pulses After Run (count)": 11525,
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
    assert "RHEED Adjustment" in html

    # Preparation sub-sections
    assert "Pre-Annealing" in html
    assert "Cool Down" in html

    # Deposition sub-sections
    assert "Laser" in html
    assert "Growth Temperature" in html
    assert "Pre-Ablation" in html
    assert "Ablation" in html
    assert "Pulse Tracking" in html
    assert "Additional On-Target" in html
    assert "After Run" in html

    # Cool Down only has Cooling Rate (no Temperature or Hold Time)
    assert "Post-Annealing Temperature" not in html
    assert "Post-Annealing Hold Time" not in html

    # Verify value appears in Growth Temperature sub-section
    assert "650.0" in html


def test_pulse_counts_are_exported_as_integers():
    result = coerce_numeric_values(
        {"target_1": {"Ablation Pulses (count)": "1500", "Laser Energy (mJ)": "245"}}
    )

    assert result["target_1"]["Ablation Pulses (count)"] == 1500
    assert isinstance(result["target_1"]["Ablation Pulses (count)"], int)
    assert result["target_1"]["Laser Energy (mJ)"] == 245.0


def test_form_calculates_target_and_all_laser_pulses():
    from pld_workflow.form import GenerateForm

    app = QApplication.instance() or QApplication([])
    form = GenerateForm()
    form.target_pulses_before_input[0].setValue(10_000)
    form.pre_number_pulses_input[0].setValue(100)
    form.number_pulses_input[0].setValue(900)
    form.additional_on_target_pulses_input[0].setValue(25)
    form.off_target_pulses_input[0].setValue(75)

    assert form.on_target_pulses_this_run_output[0].value() == 1_025
    assert form.target_pulses_after_output[0].value() == 11_025
    assert form.all_laser_pulses_this_run_output[0].value() == 1_100
    form.close()
    app.processEvents()


def test_template_mode_clears_run_specific_pulses():
    from pld_workflow.form import GenerateForm

    app = QApplication.instance() or QApplication([])
    form = GenerateForm()
    form._apply_info_dict(
        {
            "header": {"Growth ID": "OLD", "User Name": "User", "Chamber": "PLD-1"},
            "target_1": {
                "Target ID": "SRO-01",
                "Target Material": "SrRuO3",
                "Pre-Ablation Pulses (count)": 100,
                "Ablation Pulses (count)": 900,
                "Additional On-Target Pulses (count)": 25,
                "Off-Target Pulses (count)": 75,
                "Target Pulses Before Run (count)": 10_000,
            },
        },
        as_template=True,
    )

    assert form.growth_id_input.text() == ""
    assert form.target_id_input[0].text() == "SRO-01"
    assert form.pre_number_pulses_input[0].value() == 0
    assert form.number_pulses_input[0].value() == 0
    assert form.additional_on_target_pulses_input[0].value() == 0
    assert form.off_target_pulses_input[0].value() == 0
    assert form.target_pulses_before_input[0].value() == 0
    assert not form.pulse_history_ready[0]
    form.close()
    app.processEvents()

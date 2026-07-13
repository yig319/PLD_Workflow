"""PyQt form for recording PLD growth parameters.

This module supports parameter capture plus JSON and HTML export.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .parameter_export import build_default_file_stem, save_parameters_json_and_html


def resolve_header_timestamp(
    header: Dict[str, Any],
    now: datetime.datetime | None = None,
) -> tuple[str, str]:
    """Resolve header date/time values, defaulting to the current timestamp."""
    current = now or datetime.datetime.now()
    date_text = str(header.get("Date", header.get("date", ""))).strip()
    time_text = str(header.get("time", header.get("Time", ""))).strip()
    return (
        date_text or current.strftime("%m/%d/%Y"),
        time_text or current.strftime("%H:%M:%S"),
    )


class MessageWindow(QWidget):
    """Small popup used to show short status messages to the user."""

    def __init__(self, message: str):
        super().__init__()
        self.message = f"\n    {message}"
        self._init_ui()

    def _init_ui(self) -> None:
        QLabel(self.message, self)


class GenerateForm(QWidget):
    """Interactive PLD parameter form with dynamic target pages."""

    INITIAL_TARGET_COUNT = 1

    def __init__(self, version: str = "parameter"):
        super().__init__()
        if version != "parameter":
            raise ValueError("Only version='parameter' is supported.")

        self.version = version
        self.current_page = 0
        self.window_height = 22
        self.button_height = 28

        self._init_inputs()
        self._init_layout()
        self._reset_targets(self.INITIAL_TARGET_COUNT)

    def _new_line_edit(self) -> QLineEdit:
        """Create a line edit used by target-page fields."""
        return QLineEdit()

    def _new_gas_combo(self) -> QComboBox:
        """Create an editable gas combo box with common options."""
        combo = QComboBox()
        combo.addItems(["", "Vacuum", "Oxygen", "Argon"])
        combo.setEditable(True)
        return combo

    @staticmethod
    def _compact_form_layout() -> QFormLayout:
        """Create a compact form layout used inside section cards."""
        layout = QFormLayout()
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        return layout

    @staticmethod
    def _build_form_group(title: str) -> tuple[QGroupBox, QFormLayout]:
        """Create a titled group box with a compact form layout."""
        group = QGroupBox(title)
        layout = GenerateForm._compact_form_layout()
        group.setLayout(layout)
        return group, layout

    @staticmethod
    def _section_note(text: str) -> QLabel:
        """Create one subtle note label for section guidance."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setProperty("role", "section-note")
        return label

    def _init_target_storage(self) -> None:
        """Initialize parallel lists storing per-target widgets."""
        self.target_input: List[QLineEdit] = []

        self.offset_x_input: List[QLineEdit] = []
        self.offset_y_input: List[QLineEdit] = []
        self.scan_diameter_input: List[QLineEdit] = []
        self.scan_speed_xz_input: List[QLineEdit] = []

        self.heater_position_x_input: List[QLineEdit] = []
        self.heater_position_y_input: List[QLineEdit] = []
        self.heater_position_z_input: List[QLineEdit] = []
        self.heater_tilt_input: List[QLineEdit] = []
        self.heater_azimuth_input: List[QLineEdit] = []

        self.fluence_input: List[QLineEdit] = []
        self.laser_energy_input: List[QLineEdit] = []
        self.laser_voltage_input: List[QLineEdit] = []
        self.measured_energy_input: List[QLineEdit] = []

        self.mask_width_input: List[QLineEdit] = []
        self.mask_height_input: List[QLineEdit] = []
        self.mask_area_input: List[QLineEdit] = []

        self.spot_width_input: List[QLineEdit] = []
        self.spot_height_input: List[QLineEdit] = []
        self.spot_area_input: List[QLineEdit] = []
        self.magnification_input: List[QLineEdit] = []

        self.pre_number_pulses_input: List[QLineEdit] = []
        self.pre_frequency_input: List[QLineEdit] = []
        self.pre_annealing_temperature_input: List[QLineEdit] = []
        self.pre_annealing_heating_speed_input: List[QLineEdit] = []
        self.pre_annealing_time_input: List[QLineEdit] = []
        self.pre_annealing_pressure_mbar_input: List[QLineEdit] = []
        self.pre_annealing_pressure_input: List[QLineEdit] = []
        self.pre_annealing_growth_rate_input: List[QLineEdit] = []
        self.pre_annealing_growth_temperature_input: List[QLineEdit] = []
        self.growth_pressure_mbar_input: List[QLineEdit] = []
        self.growth_pressure_input: List[QLineEdit] = []
        self.post_annealing_temperature_input: List[QLineEdit] = []
        self.post_annealing_time_input: List[QLineEdit] = []
        self.post_annealing_cool_rate_input: List[QLineEdit] = []
        self.post_annealing_pressure_mbar_input: List[QLineEdit] = []
        self.post_annealing_pressure_input: List[QLineEdit] = []

        self.temperature_input: List[QLineEdit] = []
        self.gas_input: List[QComboBox] = []
        self.frequency_input: List[QLineEdit] = []
        self.number_pulses_input: List[QLineEdit] = []

    def _append_target_fields(self) -> int:
        """Append one set of target-page widgets and return its index."""
        self.target_input.append(self._new_line_edit())

        self.offset_x_input.append(self._new_line_edit())
        self.offset_y_input.append(self._new_line_edit())
        self.scan_diameter_input.append(self._new_line_edit())
        self.scan_speed_xz_input.append(self._new_line_edit())

        self.heater_position_x_input.append(self._new_line_edit())
        self.heater_position_y_input.append(self._new_line_edit())
        self.heater_position_z_input.append(self._new_line_edit())
        self.heater_tilt_input.append(self._new_line_edit())
        self.heater_azimuth_input.append(self._new_line_edit())

        self.fluence_input.append(self._new_line_edit())
        self.laser_energy_input.append(self._new_line_edit())
        self.laser_voltage_input.append(self._new_line_edit())
        self.measured_energy_input.append(self._new_line_edit())

        self.mask_width_input.append(self._new_line_edit())
        self.mask_height_input.append(self._new_line_edit())
        self.mask_area_input.append(self._new_line_edit())

        self.spot_width_input.append(self._new_line_edit())
        self.spot_height_input.append(self._new_line_edit())
        self.spot_area_input.append(self._new_line_edit())
        self.magnification_input.append(self._new_line_edit())

        self.pre_number_pulses_input.append(self._new_line_edit())
        self.pre_frequency_input.append(self._new_line_edit())
        self.pre_annealing_temperature_input.append(self._new_line_edit())
        self.pre_annealing_heating_speed_input.append(self._new_line_edit())
        self.pre_annealing_time_input.append(self._new_line_edit())
        self.pre_annealing_pressure_mbar_input.append(self._new_line_edit())
        self.pre_annealing_pressure_input.append(self._new_line_edit())
        self.pre_annealing_growth_rate_input.append(self._new_line_edit())
        self.pre_annealing_growth_temperature_input.append(self._new_line_edit())
        self.growth_pressure_mbar_input.append(self._new_line_edit())
        self.growth_pressure_input.append(self._new_line_edit())
        self.post_annealing_temperature_input.append(self._new_line_edit())
        self.post_annealing_time_input.append(self._new_line_edit())
        self.post_annealing_cool_rate_input.append(self._new_line_edit())
        self.post_annealing_pressure_mbar_input.append(self._new_line_edit())
        self.post_annealing_pressure_input.append(self._new_line_edit())

        self.temperature_input.append(self._new_line_edit())
        self.gas_input.append(self._new_gas_combo())
        self.frequency_input.append(self._new_line_edit())
        self.number_pulses_input.append(self._new_line_edit())

        return len(self.target_input) - 1

    def _init_inputs(self) -> None:
        """Create and prefill all input controls used by the form."""
        default_date, default_time = resolve_header_timestamp({})
        self.growth_id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.date_input = QLineEdit(default_date)
        self.time_input = QLineEdit(default_time)
        self.save_path_input = QLineEdit(os.getcwd())
        self.button_choose_directory = QPushButton("Browse...")
        self.button_choose_directory.setFixedHeight(self.button_height)
        self.button_choose_directory.clicked.connect(self.choose_directory)

        self.chamber_ComboBox = QComboBox()
        self.chamber_ComboBox.addItems(["PLD-1", "PLD-2", "PLD-3 TSST"])
        self.chamber_ComboBox.setEditable(True)

        substrate_list = [
            "",
            "NSO (NdScO3)",
            "GSO (GdScO3)",
            "SrTiO3 (Strontium Titanate)",
            "LSAT ((LaAlO3)0.3(Sr2AlTaO6)0.7)",
            "LAO (LaAlO3)",
            "MgO (Magnesium Oxide)",
            "Si (Silicon)",
            "None",
        ]
        self.substrate_ComboBox = QComboBox()
        self.substrate_ComboBox.addItems(substrate_list)
        self.substrate_ComboBox.setEditable(True)

        self.substrate_size_ComboBox = QComboBox()
        self.substrate_size_ComboBox.addItems(["", "5 cm * 5 cm", "2.5 cm * 2.5 cm", "Irregular"])
        self.substrate_size_ComboBox.setEditable(True)

        self.cool_down_gas = QComboBox()
        self.cool_down_gas.addItems(["", "Vacuum", "Oxygen", "Argon"])
        self.cool_down_gas.setEditable(True)

        self.notes_input = QPlainTextEdit()
        self.notes_input.setMinimumHeight(self.window_height * 2)
        self.notes_input.setMaximumHeight(self.window_height * 3)
        self.notes_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._init_target_storage()

    def _init_layout(self) -> None:
        """Compose the top-level widget layout and connect UI signals."""
        self.setMinimumSize(1100, 760)
        self.resize(1280, 1080)
        self.setWindowTitle("PLD Growth Record")
        self.setStyleSheet(
            """
            QWidget {
                background: #f4f7fb;
                color: #1b2b3a;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d4deea;
                border-radius: 10px;
                margin-top: 14px;
                font-weight: 600;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #23425f;
                background: #f4f7fb;
            }
            QLineEdit, QComboBox, QPlainTextEdit, QScrollArea {
                background: #ffffff;
                border: 1px solid #c8d3df;
                border-radius: 7px;
                padding: 4px 6px;
            }
            QPushButton {
                background: #e7eef6;
                border: 1px solid #c7d2df;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton[role="primary"] {
                background: #2b6a93;
                color: #ffffff;
                border: 1px solid #245977;
                font-weight: 600;
            }
            QLabel[role="section-note"] {
                color: #5a6e85;
            }
            """
        )

        self.toplayout = QVBoxLayout()
        self.toplayout.setSpacing(12)
        self.setLayout(self.toplayout)

        self.pageCombo = QComboBox()
        self.pageCombo.setMinimumWidth(180)
        self.pageCombo.activated.connect(self.switchPage)

        self.button_add_target = QPushButton(self)
        self.button_add_target.setText("Add Target")
        self.button_add_target.setFixedHeight(self.button_height + 4)
        self.button_add_target.clicked.connect(self.add_target)

        self.button_load = QPushButton(self)
        self.button_load.setText("Load JSON")
        self.button_load.setProperty("role", "primary")
        self.button_load.setFixedHeight(self.button_height + 6)
        self.button_load.setToolTip(
            "Load a template or saved JSON file. Missing Date and Time values default to the current timestamp."
        )
        self.button_load.clicked.connect(self.load)

        self.button_save = QPushButton(self)
        self.button_save.setText("Save Parameters")
        self.button_save.setFixedHeight(self.button_height + 6)
        self.button_save.clicked.connect(self.save)

        self.setAcceptDrops(True)

        top_panel = QWidget()
        top_panel.setProperty("role", "top-panel")
        top_panel_layout = QGridLayout()
        top_panel_layout.setContentsMargins(0, 0, 0, 0)
        top_panel_layout.setHorizontalSpacing(10)
        top_panel_layout.setVerticalSpacing(8)
        top_panel.setLayout(top_panel_layout)
        self.toplayout.addWidget(top_panel)

        top_actions = QWidget()
        top_actions_layout = QVBoxLayout()
        top_actions_layout.setContentsMargins(0, 0, 0, 0)
        top_actions_layout.setSpacing(8)
        top_actions.setLayout(top_actions_layout)
        top_actions_layout.addWidget(self.button_load)
        top_actions_layout.addWidget(self.button_save)
        top_actions_layout.addStretch(1)

        top_panel_layout.addWidget(QLabel("Growth ID"), 0, 0)
        top_panel_layout.addWidget(self.growth_id_input, 0, 1)
        top_panel_layout.addWidget(QLabel("Date"), 0, 2)
        top_panel_layout.addWidget(self.date_input, 0, 3)
        top_panel_layout.addWidget(QLabel("Substrate"), 0, 4)
        top_panel_layout.addWidget(self.substrate_ComboBox, 0, 5)
        top_panel_layout.addWidget(top_actions, 0, 6, 3, 1)

        top_panel_layout.addWidget(QLabel("User Name"), 1, 0)
        top_panel_layout.addWidget(self.name_input, 1, 1)
        top_panel_layout.addWidget(QLabel("Time"), 1, 2)
        top_panel_layout.addWidget(self.time_input, 1, 3)
        top_panel_layout.addWidget(QLabel("Substrate Size"), 1, 4)
        top_panel_layout.addWidget(self.substrate_size_ComboBox, 1, 5)

        top_panel_layout.addWidget(QLabel("Chamber"), 2, 0)
        top_panel_layout.addWidget(self.chamber_ComboBox, 2, 1)
        top_panel_layout.addWidget(QLabel("Directory"), 2, 2)
        top_panel_layout.addWidget(self._directory_row(), 2, 3, 1, 3)

        for column, stretch in enumerate([0, 1, 0, 1, 0, 1, 0]):
            top_panel_layout.setColumnStretch(column, stretch)

        self.form_notes = QGroupBox("Notes")
        self.notes_layout = QFormLayout()
        self.form_notes.setLayout(self.notes_layout)
        self.notes_layout.addRow(self.notes_input)

        self.shared_container: QWidget | None = None
        self.target_material_stack = QStackedWidget(self)
        self.Stack = QStackedWidget(self)

        self.status_box = QGroupBox("Status")
        status_box_layout = QVBoxLayout()
        status_box_layout.setContentsMargins(8, 8, 8, 8)
        self.status_box.setLayout(status_box_layout)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("role", "section-note")
        status_box_layout.addWidget(self.status_label)
        status_box_layout.addStretch()

    @staticmethod
    def _area_row(width_widget: QLineEdit, height_widget: QLineEdit, area_widget: QLineEdit) -> QWidget:
        """Create one row widget for width/height/area inputs."""
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel("W (mm)"))
        row_layout.addWidget(width_widget)
        row_layout.addWidget(QLabel("H (mm)"))
        row_layout.addWidget(height_widget)
        row_layout.addWidget(QLabel("Area (mm^2)"))
        row_layout.addWidget(area_widget)
        row_widget.setLayout(row_layout)
        return row_widget

    @staticmethod
    def _xy_row(x_widget: QLineEdit, y_widget: QLineEdit, unit: str = "") -> QWidget:
        """Create one row widget for X/Y paired inputs."""
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        x_label = f"X ({unit})" if unit else "X"
        y_label = f"Y ({unit})" if unit else "Y"
        row_layout.addWidget(QLabel(x_label))
        row_layout.addWidget(x_widget)
        row_layout.addWidget(QLabel(y_label))
        row_layout.addWidget(y_widget)
        row_widget.setLayout(row_layout)
        return row_widget

    def _pressure_row(self, mbar_widget: QLineEdit, mtorr_widget: QLineEdit) -> QWidget:
        """Create a compact pressure editor with stacked unit rows plus conversion."""
        row_widget = QWidget()
        row_layout = QGridLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setHorizontalSpacing(6)
        row_layout.setVerticalSpacing(4)

        convert_button = QPushButton("Convert")
        convert_button.setFixedHeight(self.button_height)
        convert_button.clicked.connect(lambda: self._convert_pressure_pair(mbar_widget, mtorr_widget))

        row_layout.addWidget(QLabel("mbar"), 0, 0)
        row_layout.addWidget(mbar_widget, 0, 1)
        row_layout.addWidget(QLabel("mTorr"), 1, 0)
        row_layout.addWidget(mtorr_widget, 1, 1)
        row_layout.addWidget(convert_button, 0, 2, 2, 1)
        row_layout.setColumnStretch(1, 1)
        row_widget.setLayout(row_layout)
        return row_widget

    @staticmethod
    def _convert_pressure_pair(mbar_widget: QLineEdit, mtorr_widget: QLineEdit) -> None:
        """Convert pressure units, preferring mbar to mTorr when both are present."""
        mbar_text = mbar_widget.text().strip()
        mtorr_text = mtorr_widget.text().strip()
        mbar_to_mtorr = 750.061683

        try:
            if mbar_text:
                mtorr_widget.setText(GenerateForm._format_pressure(float(mbar_text) * mbar_to_mtorr))
            elif mtorr_text:
                mbar_widget.setText(GenerateForm._format_pressure(float(mtorr_text) / mbar_to_mtorr))
        except ValueError:
            return

    @staticmethod
    def _format_pressure(value: float) -> str:
        """Format converted pressure without noisy trailing zeroes."""
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _directory_row(self) -> QWidget:
        """Create one row widget for directory path and browse button."""
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.save_path_input)
        row_layout.addWidget(self.button_choose_directory)
        row_widget.setLayout(row_layout)
        return row_widget

    def create_header(self) -> QFormLayout:
        """Build the header section containing metadata and output path."""
        header_layout = QFormLayout()
        header_layout.addRow(QLabel("Growth ID"), self.growth_id_input)
        header_layout.addRow(QLabel("User Name"), self.name_input)
        header_layout.addRow(QLabel("Date"), self.date_input)
        header_layout.addRow(QLabel("Time"), self.time_input)
        header_layout.addRow(QLabel("Directory"), self._directory_row())
        return header_layout

    def _clear_shared_sections(self) -> None:
        """Remove the shared form body before rebuilding target storage."""
        for shared_widget in (self.pageCombo, self.button_add_target, self.target_material_stack, self.Stack):
            if shared_widget.parentWidget() is not None:
                shared_widget.setParent(None)
        if self.form_notes.parentWidget() is not None:
            self.form_notes.setParent(None)
        if self.cool_down_gas.parentWidget() is not None:
            self.cool_down_gas.setParent(None)
        if self.status_box.parentWidget() is not None:
            self.status_box.setParent(None)

        if self.shared_container is not None:
            self.toplayout.removeWidget(self.shared_container)
            self.shared_container.setParent(None)
            self.shared_container.deleteLater()
            self.shared_container = None

    def _build_shared_sections(self) -> None:
        """Build the one-copy shared instrument and preparation sections."""
        self._clear_shared_sections()
        shared_index = 0

        self.shared_container = QWidget()
        shared_layout = QVBoxLayout()
        shared_layout.setContentsMargins(0, 0, 0, 0)
        self.shared_container.setLayout(shared_layout)
        self.toplayout.addWidget(self.shared_container)

        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        shared_layout.addWidget(page_scroll)

        page_body = QWidget()
        page_scroll.setWidget(page_body)

        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        page_body.setLayout(body_layout)

        instrument_group = QGroupBox("Instrument")
        instrument_layout = QVBoxLayout()
        instrument_group.setLayout(instrument_layout)
        instrument_grid = QGridLayout()
        instrument_grid.setHorizontalSpacing(12)
        instrument_grid.setVerticalSpacing(12)
        instrument_layout.addLayout(instrument_grid)

        form_target_parameters, layout_target_parameters = self._build_form_group("Target Setup")
        layout_target_parameters.addRow(
            QLabel("Offset (mm)"),
            self._xy_row(self.offset_x_input[shared_index], self.offset_y_input[shared_index]),
        )
        layout_target_parameters.addRow(QLabel("Scan Diameter (mm)"), self.scan_diameter_input[shared_index])
        layout_target_parameters.addRow(QLabel("Scan Speed X/Z (mm/s)"), self.scan_speed_xz_input[shared_index])

        form_heater, layout_heater = self._build_form_group("Heater Position")
        layout_heater.addRow(QLabel("Position X (mm)"), self.heater_position_x_input[shared_index])
        layout_heater.addRow(QLabel("Position Y (mm)"), self.heater_position_y_input[shared_index])
        layout_heater.addRow(QLabel("Position Z (mm)"), self.heater_position_z_input[shared_index])

        form_mask_spot, layout_mask_spot = self._build_form_group("Mask and Spot")
        layout_mask_spot.addRow(QLabel("Mask Area"), self.mask_area_input[shared_index])
        layout_mask_spot.addRow(QLabel("Spot Area"), self.spot_area_input[shared_index])
        layout_mask_spot.addRow(QLabel("Magnification"), self.magnification_input[shared_index])

        form_rheed, layout_rheed = self._build_form_group("RHEED Adjustment")
        layout_rheed.addRow(QLabel("Tilt (deg)"), self.heater_tilt_input[shared_index])
        layout_rheed.addRow(QLabel("Azimuth (deg)"), self.heater_azimuth_input[shared_index])

        instrument_grid.addWidget(form_target_parameters, 0, 0)
        instrument_grid.addWidget(form_heater, 0, 1)
        instrument_grid.addWidget(form_mask_spot, 0, 2)
        instrument_grid.addWidget(form_rheed, 0, 3)
        instrument_grid.setColumnStretch(0, 3)
        instrument_grid.setColumnStretch(1, 2)
        instrument_grid.setColumnStretch(2, 2)
        instrument_grid.setColumnStretch(3, 2)
        body_layout.addWidget(instrument_group)

        process_group = QGroupBox("Preparation")
        process_layout = QVBoxLayout()
        process_group.setLayout(process_layout)
        process_grid = QGridLayout()
        process_grid.setHorizontalSpacing(12)
        process_grid.setVerticalSpacing(12)
        process_layout.addLayout(process_grid)

        form_pre_annealing = QGroupBox("Pre-Annealing")
        pre_grid = QGridLayout()
        pre_grid.setContentsMargins(10, 8, 10, 10)
        pre_grid.setHorizontalSpacing(12)
        pre_grid.setVerticalSpacing(6)
        form_pre_annealing.setLayout(pre_grid)
        pre_left = self._compact_form_layout()
        pre_right = self._compact_form_layout()
        pre_grid.addLayout(pre_left, 0, 0)
        pre_grid.addLayout(pre_right, 0, 1)
        pre_grid.setColumnStretch(0, 1)
        pre_grid.setColumnStretch(1, 1)
        pre_left.addRow(
            QLabel("Heat Rate (\N{DEGREE SIGN}C/min)"),
            self.pre_annealing_heating_speed_input[shared_index],
        )
        pre_left.addRow(
            QLabel("Anneal Temperature (\N{DEGREE SIGN}C)"),
            self.pre_annealing_temperature_input[shared_index],
        )
        pre_left.addRow(
            QLabel("Hold Time (min)"),
            self.pre_annealing_time_input[shared_index],
        )
        pre_right.addRow(
            QLabel("Pressure"),
            self._pressure_row(
                self.pre_annealing_pressure_mbar_input[shared_index],
                self.pre_annealing_pressure_input[shared_index],
            ),
        )

        form_post_annealing = QGroupBox("Cool Down")
        post_grid = QGridLayout()
        post_grid.setContentsMargins(10, 8, 10, 10)
        post_grid.setHorizontalSpacing(12)
        post_grid.setVerticalSpacing(6)
        form_post_annealing.setLayout(post_grid)
        post_left = self._compact_form_layout()
        post_right = self._compact_form_layout()
        post_grid.addLayout(post_left, 0, 0)
        post_grid.addLayout(post_right, 0, 1)
        post_grid.setColumnStretch(0, 1)
        post_grid.setColumnStretch(1, 1)
        post_left.addRow(
            QLabel("Cooling Rate (\N{DEGREE SIGN}C/min)"),
            self.post_annealing_cool_rate_input[shared_index],
        )
        post_left.addRow(QLabel("Atmosphere"), self.cool_down_gas)
        post_right.addRow(
            QLabel("Pressure"),
            self._pressure_row(
                self.post_annealing_pressure_mbar_input[shared_index],
                self.post_annealing_pressure_input[shared_index],
            ),
        )

        deposition_group = QGroupBox("Deposition")
        deposition_layout = QGridLayout()
        deposition_layout.setContentsMargins(10, 8, 10, 10)
        deposition_layout.setHorizontalSpacing(12)
        deposition_layout.setVerticalSpacing(8)
        deposition_group.setLayout(deposition_layout)
        deposition_layout.addWidget(self.pageCombo, 0, 0)
        deposition_layout.addWidget(self.target_material_stack, 0, 1)
        deposition_layout.addWidget(self.button_add_target, 0, 2)
        deposition_layout.addWidget(self.Stack, 1, 0, 1, 3)
        deposition_layout.setColumnStretch(0, 1)
        deposition_layout.setColumnStretch(1, 1)
        deposition_layout.setColumnStretch(2, 0)

        process_grid.addWidget(form_pre_annealing, 0, 0)
        process_grid.addWidget(form_post_annealing, 0, 1)
        process_grid.setColumnStretch(0, 1)
        process_grid.setColumnStretch(1, 1)
        body_layout.addWidget(process_group)
        body_layout.addWidget(deposition_group)
        notes_row = QHBoxLayout()
        notes_row.addWidget(self.form_notes, 2)
        notes_row.addWidget(self.status_box, 1)
        body_layout.addLayout(notes_row)
        body_layout.addStretch(1)

    def stackUI(self, create_index: int) -> QVBoxLayout:
        """Build the per-target deposition controls only."""
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        deposition_grid = QGridLayout()
        deposition_grid.setContentsMargins(0, 0, 0, 0)
        deposition_grid.setHorizontalSpacing(12)
        deposition_grid.setVerticalSpacing(8)
        page_layout.addLayout(deposition_grid)

        form_laser = QGroupBox("Laser")
        laser_grid = QGridLayout()
        laser_grid.setContentsMargins(10, 8, 10, 10)
        laser_grid.setHorizontalSpacing(10)
        laser_grid.setVerticalSpacing(6)
        form_laser.setLayout(laser_grid)
        laser_left = self._compact_form_layout()
        laser_right = self._compact_form_layout()
        laser_grid.addLayout(laser_left, 0, 0)
        laser_grid.addLayout(laser_right, 0, 1)
        laser_grid.setColumnStretch(0, 1)
        laser_grid.setColumnStretch(1, 1)
        laser_left.addRow(QLabel("Voltage (kV)"), self.laser_voltage_input[create_index])
        laser_left.addRow(QLabel("Fluence (J/cm^2)"), self.fluence_input[create_index])
        laser_right.addRow(QLabel("Energy (mJ)"), self.laser_energy_input[create_index])
        laser_right.addRow(
            QLabel("Measured Energy (mJ)"),
            self.measured_energy_input[create_index],
        )

        form_growth_temp, layout_growth_temp = self._build_form_group("Growth Temperature")
        layout_growth_temp.addRow(
            QLabel("Heating Rate (\N{DEGREE SIGN}C/min)"),
            self.pre_annealing_growth_rate_input[create_index],
        )
        layout_growth_temp.addRow(
            QLabel("Temperature (\N{DEGREE SIGN}C)"),
            self.temperature_input[create_index],
        )

        form_atmosphere, layout_atmosphere = self._build_form_group("Atmosphere")
        layout_atmosphere.addRow(QLabel("Atmosphere"), self.gas_input[create_index])
        layout_atmosphere.addRow(
            QLabel("Pressure"),
            self._pressure_row(
                self.growth_pressure_mbar_input[create_index],
                self.growth_pressure_input[create_index],
            ),
        )

        form_pre_ablation, layout_pre_ablation = self._build_form_group("Pre-Ablation")
        layout_pre_ablation.addRow(QLabel("Frequency (Hz)"), self.pre_frequency_input[create_index])
        layout_pre_ablation.addRow(QLabel("Pulses (count)"), self.pre_number_pulses_input[create_index])

        form_ablation, layout_ablation = self._build_form_group("Ablation")
        layout_ablation.addRow(QLabel("Frequency (Hz)"), self.frequency_input[create_index])
        layout_ablation.addRow(QLabel("Pulses (count)"), self.number_pulses_input[create_index])

        for form in (form_laser, form_growth_temp, form_atmosphere, form_pre_ablation, form_ablation):
            form.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        deposition_grid.addWidget(form_laser, 0, 0, 1, 2)
        deposition_grid.addWidget(form_growth_temp, 0, 2)
        deposition_grid.addWidget(form_atmosphere, 1, 0)
        deposition_grid.addWidget(form_pre_ablation, 1, 1)
        deposition_grid.addWidget(form_ablation, 1, 2)
        deposition_grid.setColumnStretch(0, 1)
        deposition_grid.setColumnStretch(1, 1)
        deposition_grid.setColumnStretch(2, 1)

        return page_layout

    def add_target(self, set_current: bool = True) -> None:
        """Add one new target page and its corresponding widgets."""
        idx = self._append_target_fields()

        target_material_page = QWidget()
        target_material_layout = QHBoxLayout()
        target_material_layout.setContentsMargins(0, 0, 0, 0)
        target_material_layout.setSpacing(8)
        target_material_page.setLayout(target_material_layout)
        target_material_layout.addWidget(QLabel("Target Material"))
        target_material_layout.addWidget(self.target_input[idx])
        self.target_material_stack.addWidget(target_material_page)

        page = QWidget()
        page.setLayout(self.stackUI(idx))
        self.Stack.addWidget(page)
        self.pageCombo.addItem(f"Target_{idx + 1}")

        if set_current:
            self.pageCombo.setCurrentIndex(idx)
            self.switchPage(idx)

    def _reset_targets(self, count: int) -> None:
        """Reset all target pages to exactly `count` pages."""
        target_count = max(1, int(count))

        self._clear_shared_sections()

        while self.target_material_stack.count() > 0:
            page = self.target_material_stack.widget(0)
            self.target_material_stack.removeWidget(page)
            page.deleteLater()

        while self.Stack.count() > 0:
            page = self.Stack.widget(0)
            self.Stack.removeWidget(page)
            page.deleteLater()

        self.pageCombo.clear()
        self._init_target_storage()

        for _ in range(target_count):
            self.add_target(set_current=False)

        self._build_shared_sections()
        self.pageCombo.setCurrentIndex(0)
        self.switchPage(0)

    def show_message_window(self, message: str) -> None:
        """Display a transient popup with operation status text."""
        self.exPopup = MessageWindow(message)
        self.exPopup.setGeometry(500, 500, 450, 100)
        self.exPopup.show()

    def switchPage(self, index: int) -> None:
        """Switch the active deposition controls for the selected target."""
        self.target_material_stack.setCurrentIndex(index)
        self.Stack.setCurrentIndex(index)
        self.current_page = index

    def choose_directory(self) -> None:
        """Open a folder dialog and set the save directory field."""
        start_dir = self.save_path_input.text().strip() or os.getcwd()
        selected = QFileDialog.getExistingDirectory(self, "Select Save Directory", start_dir)
        if selected:
            self.save_path_input.setText(selected)

    @staticmethod
    def _set_line_edit_value(widget: QLineEdit, value: Any) -> None:
        """Set line edit text while safely converting values to string."""
        widget.setText("" if value is None else str(value))

    @staticmethod
    def _set_combo_value(widget: QComboBox, value: Any) -> None:
        """Set combo text, adding value as option when needed."""
        text = "" if value is None else str(value)
        if text and widget.findText(text) < 0:
            widget.addItem(text)
        widget.setCurrentText(text)

    @staticmethod
    def _first_present_value(data: Dict[str, Any], *keys: str) -> Any:
        """Return first existing key value from a mapping, else None."""
        for key in keys:
            if key in data:
                return data.get(key)
        return None

    def _apply_info_dict(self, info_dict: Dict[str, Dict[str, Any]], source_path: str | None = None) -> None:
        """Populate form fields from a dictionary loaded from JSON."""
        header = info_dict.get("header", {})
        resolved_date, resolved_time = resolve_header_timestamp(header)
        resolved_path = str(Path(source_path).resolve().parent) if source_path else self.save_path_input.text()

        self._set_line_edit_value(self.growth_id_input, header.get("Growth ID"))
        self._set_line_edit_value(self.name_input, header.get("User Name"))
        self._set_line_edit_value(self.date_input, resolved_date)
        self._set_line_edit_value(self.time_input, resolved_time)
        self._set_line_edit_value(self.save_path_input, header.get("Path") or resolved_path)
        self._set_combo_value(self.chamber_ComboBox, header.get("Chamber"))
        self._set_combo_value(
            self.cool_down_gas,
            self._first_present_value(header, "Cool Down Atmosphere", "Post-Annealing Atmosphere"),
        )
        self.notes_input.setPlainText(str(header.get("Notes", "")))

        substrate = header.get("Substrate")
        if not substrate:
            for key in ("Substrate_1", "Substrate_2", "Substrate_3", "Substrate_4"):
                substrate = header.get(key)
                if substrate:
                    break
        self._set_combo_value(self.substrate_ComboBox, substrate)
        self._set_combo_value(self.substrate_size_ComboBox, header.get("Substrate Size"))

        target_keys = [k for k in info_dict.keys() if k.startswith("target_")]
        target_keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 10**9)

        if not target_keys:
            self._reset_targets(1)
            target_keys = ["target_1"]
        else:
            self._reset_targets(len(target_keys))

        for i, key in enumerate(target_keys):
            target_dict = info_dict.get(key, {})
            self._set_line_edit_value(self.target_input[i], target_dict.get("Target Material"))

            self._set_line_edit_value(
                self.offset_x_input[i],
                self._first_present_value(target_dict, "Offset X (mm)", "Offset X"),
            )
            self._set_line_edit_value(
                self.offset_y_input[i],
                self._first_present_value(target_dict, "Offset Y (mm)", "Offset Y"),
            )
            self._set_line_edit_value(
                self.scan_diameter_input[i],
                self._first_present_value(target_dict, "Scan Diameter (mm)", "Scan Diameter"),
            )
            self._set_line_edit_value(
                self.scan_speed_xz_input[i],
                self._first_present_value(target_dict, "Scan Speed X/Z (mm/s)"),
            )

            self._set_line_edit_value(
                self.heater_position_x_input[i],
                self._first_present_value(target_dict, "Heater Position X (mm)", "Heater Position X"),
            )
            self._set_line_edit_value(
                self.heater_position_y_input[i],
                self._first_present_value(target_dict, "Heater Position Y (mm)", "Heater Position Y"),
            )
            self._set_line_edit_value(
                self.heater_position_z_input[i],
                self._first_present_value(target_dict, "Heater Position Z (mm)", "Heater Position Z"),
            )
            self._set_line_edit_value(
                self.heater_tilt_input[i],
                self._first_present_value(target_dict, "Tilt (deg)", "Tilt", "Tile"),
            )
            self._set_line_edit_value(
                self.heater_azimuth_input[i],
                self._first_present_value(target_dict, "Azimuth (deg)", "Azimuth"),
            )

            self._set_line_edit_value(
                self.fluence_input[i],
                self._first_present_value(target_dict, "Fluence (J/cm^2)", "Fluence"),
            )
            
            self._set_line_edit_value(
                self.laser_energy_input[i],
                self._first_present_value(target_dict, "Laser Energy (mJ)", "Laser Energy(mJ)"),
            )
            self._set_line_edit_value(
                self.laser_voltage_input[i],
                self._first_present_value(target_dict, "Laser Voltage (kV)", "Laser Voltage(kV)"),
            )
            self._set_line_edit_value(
                self.measured_energy_input[i],
                self._first_present_value(
                    target_dict,
                    "Measured Energy (mJ)",
                    "Measured Energy(mJ)",
                    "Measured Energy Mean(mJ)",
                ),
            )


            self._set_line_edit_value(
                self.mask_width_input[i],
                self._first_present_value(target_dict, "Mask Width (mm)", "Mask Width"),
            )
            self._set_line_edit_value(
                self.mask_height_input[i],
                self._first_present_value(target_dict, "Mask Height (mm)", "Mask Height"),
            )
            self._set_line_edit_value(
                self.mask_area_input[i],
                self._first_present_value(target_dict, "Mask Area (mm^2)", "Mask Area"),
            )

            self._set_line_edit_value(
                self.spot_width_input[i],
                self._first_present_value(target_dict, "Spot Width (mm)", "Spot Width"),
            )
            self._set_line_edit_value(
                self.spot_height_input[i],
                self._first_present_value(target_dict, "Spot Height (mm)", "Spot Height"),
            )
            self._set_line_edit_value(
                self.spot_area_input[i],
                self._first_present_value(target_dict, "Spot Area (mm^2)", "Spot Area"),
            )
            self._set_line_edit_value(
                self.magnification_input[i],
                self._first_present_value(target_dict, "Magnification (x)", "Magnification"),
            )

            self._set_line_edit_value(
                self.pre_number_pulses_input[i],
                self._first_present_value(target_dict, "Pre-Ablation Pulses (count)", "Pre-Ablation-Pulses"),
            )
            self._set_line_edit_value(
                self.pre_frequency_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Ablation Frequency (Hz)",
                    "Pre-Ablation-Frequency(Hz)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Temperature (\N{DEGREE SIGN}C)",
                    "Pre-Annealing-Temperature(\N{DEGREE SIGN}C)",
                    "Pre-Annealing Temperature (degC)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_heating_speed_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Heat Rate (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing-Heating-Speed(\N{DEGREE SIGN}C/min)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_time_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Hold Time (min)",
                    "Pre-Annealing Time (min)",
                    "Pre-Annealing-Time(min)",
                    "Pre-Annealing-Time",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_growth_rate_input[i],
                self._first_present_value(
                    target_dict,
                    "Growth Condition Heating Rate (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Heating Rate (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Cool/Heat Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Cooling Rate to Growth Temp (\N{DEGREE SIGN}C/min)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_growth_temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Growth Temperature (\N{DEGREE SIGN}C)",
                    "Pre-Annealing Growth Temperature (degC)",
                    "Growth Temperature (\N{DEGREE SIGN}C)",
                    "Growth Temperature (degC)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_pressure_mbar_input[i],
                self._first_present_value(target_dict, "Pre-Annealing Pressure (mbar)"),
            )
            self._set_line_edit_value(
                self.pre_annealing_pressure_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Pressure (mTorr)",
                    "Pre-Annealing Atmosphere Pressure (mTorr)",
                    "Pre-Annealing-Atmosphere-Pressure(mTorr)",
                ),
            )
            self._set_line_edit_value(
                self.growth_pressure_mbar_input[i],
                self._first_present_value(
                    target_dict,
                    "Growth Pressure (mbar)",
                    "Ablation Pressure (mbar)",
                ),
            )
            self._set_line_edit_value(
                self.growth_pressure_input[i],
                self._first_present_value(
                    target_dict,
                    "Growth Pressure (mTorr)",
                    "Ablation Pressure (mTorr)",
                    "Ablation-Pressure(mTorr)",
                ),
            )
            self._set_line_edit_value(
                self.post_annealing_temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Post-Annealing Temperature (\N{DEGREE SIGN}C)",
                    "Post-Annealing-Temperature(\N{DEGREE SIGN}C)",
                    "Post-Annealing Temperature (degC)",
                ),
            )
            self._set_line_edit_value(
                self.post_annealing_time_input[i],
                self._first_present_value(
                    target_dict,
                    "Post-Annealing Hold Time (min)",
                    "Post-Annealing Time (min)",
                    "Post-Annealing-Time(min)",
                ),
            )
            self._set_line_edit_value(
                self.post_annealing_cool_rate_input[i],
                self._first_present_value(
                    target_dict,
                    "Post-Annealing Cooling Rate (\N{DEGREE SIGN}C/min)",
                    "Post-Annealing Cool Rate (\N{DEGREE SIGN}C/min)",
                ),
            )
            self._set_line_edit_value(
                self.post_annealing_pressure_mbar_input[i],
                self._first_present_value(target_dict, "Post-Annealing Pressure (mbar)"),
            )
            self._set_line_edit_value(
                self.post_annealing_pressure_input[i],
                self._first_present_value(
                    target_dict,
                    "Post-Annealing Pressure (mTorr)",
                    "Post-Annealing Atmosphere Pressure (mTorr)",
                    "Post-Annealing-Atmosphere-Pressure(mTorr)",
                ),
            )

            self._set_line_edit_value(
                self.temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Ablation Temperature (\N{DEGREE SIGN}C)",
                    "Ablation-Temperature(\N{DEGREE SIGN}C)",
                    "Ablation Temperature (degC)",
                    "Pre-Annealing Growth Temperature (\N{DEGREE SIGN}C)",
                    "Pre-Annealing Growth Temperature (degC)",
                    "Growth Temperature (\N{DEGREE SIGN}C)",
                    "Growth Temperature (degC)",
                ),
            )
            self._set_combo_value(
                self.gas_input[i],
                self._first_present_value(target_dict, "Ablation Atmosphere Gas", "Ablation-Atmosphere Gas"),
            )
            self._set_line_edit_value(
                self.frequency_input[i],
                self._first_present_value(target_dict, "Ablation Frequency (Hz)", "Ablation-Frequency(Hz)"),
            )
            self._set_line_edit_value(
                self.number_pulses_input[i],
                self._first_present_value(target_dict, "Ablation Pulses (count)", "Ablation-Pulses"),
            )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".json"):
                self._load_json_file(file_path)
                break

    def _load_json_file(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                info_dict = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"Failed to load JSON: {exc}")
            return
        self._apply_info_dict(info_dict, source_path=file_path)
        self.status_label.setText("Parameters loaded!")

    def load(self) -> None:
        """Load a saved JSON file and populate the form fields."""
        start_dir = self.save_path_input.text().strip() or os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Parameters JSON",
            start_dir,
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        self._load_json_file(file_path)

    def get_info(self) -> Dict[str, Dict[str, Any]]:
        """Collect all non-empty form values into a nested dictionary."""
        info_dict: Dict[str, Dict[str, Any]] = {
            "header": {
                "Growth ID": self.growth_id_input.text(),
                "User Name": self.name_input.text(),
                "Date": self.date_input.text(),
                "time": self.time_input.text(),
                "Path": self.save_path_input.text(),
                "Chamber": self.chamber_ComboBox.currentText(),
                "Substrate": self.substrate_ComboBox.currentText(),
                "Substrate Size": self.substrate_size_ComboBox.currentText(),
                "Cool Down Atmosphere": self.cool_down_gas.currentText(),
                "Notes": self.notes_input.toPlainText(),
            }
        }

        target_count = len(self.target_input)
        shared_index = 0
        for i in range(target_count):
            info_dict[f"target_{i + 1}"] = {
                "Target Material": self.target_input[i].text(),
                "Offset X (mm)": self.offset_x_input[shared_index].text(),
                "Offset Y (mm)": self.offset_y_input[shared_index].text(),
                "Scan Diameter (mm)": self.scan_diameter_input[shared_index].text(),
                "Scan Speed X/Z (mm/s)": self.scan_speed_xz_input[shared_index].text(),
                "Heater Position X (mm)": self.heater_position_x_input[shared_index].text(),
                "Heater Position Y (mm)": self.heater_position_y_input[shared_index].text(),
                "Heater Position Z (mm)": self.heater_position_z_input[shared_index].text(),
                "Tilt (deg)": self.heater_tilt_input[shared_index].text(),
                "Azimuth (deg)": self.heater_azimuth_input[shared_index].text(),
                "Fluence (J/cm^2)": self.fluence_input[i].text(),
                "Laser Energy (mJ)": self.laser_energy_input[i].text(),
                "Laser Voltage (kV)": self.laser_voltage_input[i].text(),
                "Measured Energy (mJ)": self.measured_energy_input[i].text(),
                "Mask Width (mm)": self.mask_width_input[shared_index].text(),
                "Mask Height (mm)": self.mask_height_input[shared_index].text(),
                "Mask Area (mm^2)": self.mask_area_input[shared_index].text(),
                "Spot Width (mm)": self.spot_width_input[shared_index].text(),
                "Spot Height (mm)": self.spot_height_input[shared_index].text(),
                "Spot Area (mm^2)": self.spot_area_input[shared_index].text(),
                "Magnification (x)": self.magnification_input[shared_index].text(),
                "Pre-Ablation Pulses (count)": self.pre_number_pulses_input[i].text(),
                "Pre-Ablation Frequency (Hz)": self.pre_frequency_input[i].text(),
                "Pre-Annealing Temperature (\N{DEGREE SIGN}C)": self.pre_annealing_temperature_input[
                    shared_index
                ].text(),
                "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)": self.pre_annealing_heating_speed_input[
                    shared_index
                ].text(),
                "Pre-Annealing Time (min)": self.pre_annealing_time_input[shared_index].text(),
                "Growth Condition Heating Rate (\N{DEGREE SIGN}C/min)": self.pre_annealing_growth_rate_input[
                    i
                ].text(),
                "Pre-Annealing Pressure (mbar)": self.pre_annealing_pressure_mbar_input[shared_index].text(),
                "Pre-Annealing Pressure (mTorr)": self.pre_annealing_pressure_input[
                    shared_index
                ].text(),
                "Growth Pressure (mbar)": self.growth_pressure_mbar_input[i].text(),
                "Growth Pressure (mTorr)": self.growth_pressure_input[
                    i
                ].text(),
                "Post-Annealing Cooling Rate (\N{DEGREE SIGN}C/min)": self.post_annealing_cool_rate_input[
                    shared_index
                ].text(),
                "Post-Annealing Pressure (mbar)": self.post_annealing_pressure_mbar_input[shared_index].text(),
                "Post-Annealing Pressure (mTorr)": self.post_annealing_pressure_input[
                    shared_index
                ].text(),
                "Ablation Temperature (\N{DEGREE SIGN}C)": self.temperature_input[i].text(),
                "Ablation Atmosphere Gas": self.gas_input[i].currentText(),
                "Ablation Frequency (Hz)": self.frequency_input[i].text(),
                "Ablation Pulses (count)": self.number_pulses_input[i].text(),
            }

        for section_name in list(info_dict.keys()):
            section = {k: v for k, v in info_dict[section_name].items() if v}
            if section:
                info_dict[section_name] = section
            else:
                info_dict.pop(section_name)

        return info_dict

    def _default_file_stem(self) -> str:
        """Build output file stem from core header fields."""
        return build_default_file_stem(
            self.growth_id_input.text(),
            self.name_input.text(),
            self.date_input.text(),
        )

    def save(self) -> None:
        """Serialize the current form state to JSON and HTML files."""
        try:
            result = save_parameters_json_and_html(
                info_dict=self.get_info(),
                output_dir=self.save_path_input.text().strip() or os.getcwd(),
                file_stem=self._default_file_stem(),
            )
        except OSError as exc:
            self.status_label.setText(f"Failed to save JSON: {exc}")
            return

        if result.html_error:
            self.status_label.setText(f"JSON saved. HTML export failed: {result.html_error}")
            return

        self.status_label.setText("Parameters saved to JSON and HTML!")

"""PyQt form for recording PLD growth parameters.

This module supports parameter capture plus JSON and HTML export.
"""

from __future__ import annotations

import datetime
import html
import json
import os
from typing import Any, Dict, List

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
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
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

        self.laser_voltage_input: List[QLineEdit] = []
        self.laser_energy_input: List[QLineEdit] = []
        self.targeted_measured_energy_input: List[QLineEdit] = []
        self.fluence_input: List[QLineEdit] = []

        self.mask_width_input: List[QLineEdit] = []
        self.mask_height_input: List[QLineEdit] = []
        self.mask_area_input: List[QLineEdit] = []

        self.spot_width_input: List[QLineEdit] = []
        self.spot_height_input: List[QLineEdit] = []
        self.spot_area_input: List[QLineEdit] = []
        self.magnification_input: List[QLineEdit] = []

        self.pre_number_pulses_input: List[QLineEdit] = []
        self.pre_annealing_temperature_input: List[QLineEdit] = []
        self.pre_annealing_heating_speed_input: List[QLineEdit] = []
        self.pre_annealing_time_input: List[QLineEdit] = []
        self.pre_annealing_atmosphere_pressure_input: List[QLineEdit] = []

        self.temperature_input: List[QLineEdit] = []
        self.pressure_input: List[QLineEdit] = []
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

        self.laser_voltage_input.append(self._new_line_edit())
        self.laser_energy_input.append(self._new_line_edit())
        self.targeted_measured_energy_input.append(self._new_line_edit())
        self.fluence_input.append(self._new_line_edit())

        self.mask_width_input.append(self._new_line_edit())
        self.mask_height_input.append(self._new_line_edit())
        self.mask_area_input.append(self._new_line_edit())

        self.spot_width_input.append(self._new_line_edit())
        self.spot_height_input.append(self._new_line_edit())
        self.spot_area_input.append(self._new_line_edit())
        self.magnification_input.append(self._new_line_edit())

        self.pre_number_pulses_input.append(self._new_line_edit())
        self.pre_annealing_temperature_input.append(self._new_line_edit())
        self.pre_annealing_heating_speed_input.append(self._new_line_edit())
        self.pre_annealing_time_input.append(self._new_line_edit())
        self.pre_annealing_atmosphere_pressure_input.append(self._new_line_edit())

        self.temperature_input.append(self._new_line_edit())
        self.pressure_input.append(self._new_line_edit())
        self.gas_input.append(self._new_gas_combo())
        self.frequency_input.append(self._new_line_edit())
        self.number_pulses_input.append(self._new_line_edit())

        return len(self.target_input) - 1

    def _init_inputs(self) -> None:
        """Create and prefill all input controls used by the form."""
        self.growth_id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.date_input = QLineEdit(datetime.datetime.today().strftime("%m/%d/%Y"))
        self.time_input = QLineEdit(datetime.datetime.now().strftime("%H:%M:%S"))
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
        self.substrate_size_ComboBox.addItems(["", "5 cm * 5 cm", "2.5 cm * 2.5 cm"])
        self.substrate_size_ComboBox.setEditable(True)

        self.cool_down_gas = QComboBox()
        self.cool_down_gas.addItems(["Vacuum", "Oxygen", "Argon"])
        self.cool_down_gas.setEditable(True)

        self.notes_input = QPlainTextEdit()
        self.notes_input.setMinimumHeight(self.window_height * 6)
        self.notes_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

        self._init_target_storage()

    def _init_layout(self) -> None:
        """Compose the top-level widget layout and connect UI signals."""
        self.setMinimumSize(900, 820)
        self.setWindowTitle("PLD Growth Record")

        self.toplayout = QVBoxLayout()
        self.setLayout(self.toplayout)

        self.layout = QGridLayout()
        self.toplayout.addLayout(self.layout)

        self.header_form = QGroupBox("Header")
        self.header_layout = self.create_header()
        self.header_form.setLayout(self.header_layout)
        self.layout.addWidget(self.header_form, 0, 0)

        self.chamber_form = QGroupBox("Chamber Parameters")
        self.chamber_layout = self.create_chamber()
        self.chamber_form.setLayout(self.chamber_layout)
        self.layout.addWidget(self.chamber_form, 0, 1)

        self.button_layout = QGridLayout()
        self.toplayout.addLayout(self.button_layout)

        self.pageCombo = QComboBox()
        self.pageCombo.activated.connect(self.switchPage)
        self.button_layout.addWidget(self.pageCombo, 0, 0)

        self.button_add_target = QPushButton(self)
        self.button_add_target.setText("Add Target")
        self.button_add_target.clicked.connect(self.add_target)
        self.button_layout.addWidget(self.button_add_target, 0, 1)

        self.button_load = QPushButton(self)
        self.button_load.setText("Load Parameters from JSON")
        self.button_load.clicked.connect(self.load)
        self.button_layout.addWidget(self.button_load, 0, 2)

        self.button_layout.setColumnStretch(0, 2)
        self.button_layout.setColumnStretch(1, 1)
        self.button_layout.setColumnStretch(2, 1)

        self.multiPages = QFormLayout()
        self.toplayout.addLayout(self.multiPages)

        self.Stack = QStackedWidget(self)
        self.multiPages.addWidget(self.Stack)

        self.button_save = QPushButton(self)
        self.button_save.setText("Save Parameters")
        self.button_save.clicked.connect(self.save)
        self.toplayout.addWidget(self.button_save)

        self.form_notes = QGroupBox()
        self.notes_layout = QFormLayout()
        self.form_notes.setLayout(self.notes_layout)
        self.notes_layout.addRow(QLabel("Notes"), self.notes_input)
        self.toplayout.addWidget(self.form_notes)

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

    def create_chamber(self) -> QFormLayout:
        """Build chamber-level inputs shared by all targets."""
        chamber_layout = QFormLayout()
        chamber_layout.addRow(QLabel("Chamber"), self.chamber_ComboBox)
        chamber_layout.addRow(QLabel("Substrate"), self.substrate_ComboBox)
        chamber_layout.addRow(QLabel("Substrate Size"), self.substrate_size_ComboBox)
        chamber_layout.addRow(QLabel("Cool Down Atmosphere"), self.cool_down_gas)
        return chamber_layout

    def stackUI(self, create_index: int) -> QGridLayout:
        """Build one target page for a specific target index."""
        layout = QGridLayout()

        form_target = QGroupBox()
        layout_target = QFormLayout()
        form_target.setLayout(layout_target)
        layout.addWidget(form_target, 0, 0)
        layout_target.addRow(QLabel("Target:"), self.target_input[create_index])

        form_target_parameters = QGroupBox("Target Parameters")
        layout_target_parameters = QFormLayout()
        form_target_parameters.setLayout(layout_target_parameters)
        layout.addWidget(form_target_parameters, 1, 0)
        layout_target_parameters.addRow(
            QLabel("Offset (mm)"),
            self._xy_row(self.offset_x_input[create_index], self.offset_y_input[create_index]),
        )
        layout_target_parameters.addRow(QLabel("Scan Diameter (mm)"), self.scan_diameter_input[create_index])
        layout_target_parameters.addRow(QLabel("Scan Speed X/Z (mm/s)"), self.scan_speed_xz_input[create_index])

        form_heater = QGroupBox("Heater Parameter")
        layout_heater = QFormLayout()
        form_heater.setLayout(layout_heater)
        layout.addWidget(form_heater, 1, 1)
        layout_heater.addRow(QLabel("Heater Position X (mm)"), self.heater_position_x_input[create_index])
        layout_heater.addRow(QLabel("Heater Position Y (mm)"), self.heater_position_y_input[create_index])
        layout_heater.addRow(QLabel("Heater Position Z (mm)"), self.heater_position_z_input[create_index])
        layout_heater.addRow(QLabel("Tilt (deg)"), self.heater_tilt_input[create_index])
        layout_heater.addRow(QLabel("Azimuth (deg)"), self.heater_azimuth_input[create_index])

        form_laser = QGroupBox("Laser and Mask")
        layout_laser = QFormLayout()
        form_laser.setLayout(layout_laser)
        layout.addWidget(form_laser, 1, 2)
        layout_laser.addRow(QLabel("Laser Voltage (kV)"), self.laser_voltage_input[create_index])
        layout_laser.addRow(QLabel("Laser Energy (mJ)"), self.laser_energy_input[create_index])
        layout_laser.addRow(
            QLabel("Targeted Measured Energy (mJ)"),
            self.targeted_measured_energy_input[create_index],
        )
        layout_laser.addRow(QLabel("Fluence (J/cm^2)"), self.fluence_input[create_index])

        form_chamber_target = QGroupBox("Chamber")
        layout_chamber_target = QFormLayout()
        form_chamber_target.setLayout(layout_chamber_target)
        layout.addWidget(form_chamber_target, 1, 3)
        layout_chamber_target.addRow(QLabel("Temperature (\N{DEGREE SIGN}C)"), self.temperature_input[create_index])
        layout_chamber_target.addRow(QLabel("Pressure (mTorr)"), self.pressure_input[create_index])
        layout_chamber_target.addRow(QLabel("Atmosphere Gas"), self.gas_input[create_index])

        form_mask_spot = QGroupBox("Mask and Spot")
        layout_mask_spot = QFormLayout()
        form_mask_spot.setLayout(layout_mask_spot)
        layout.addWidget(form_mask_spot, 2, 0, 1, 2)
        layout_mask_spot.addRow(
            QLabel("Mask Area"),
            self._area_row(
                self.mask_width_input[create_index],
                self.mask_height_input[create_index],
                self.mask_area_input[create_index],
            ),
        )
        layout_mask_spot.addRow(
            QLabel("Spot Area"),
            self._area_row(
                self.spot_width_input[create_index],
                self.spot_height_input[create_index],
                self.spot_area_input[create_index],
            ),
        )
        layout_mask_spot.addRow(QLabel("Magnification (x)"), self.magnification_input[create_index])

        form_pre_annealing = QGroupBox("Pre-annealing")
        layout_pre_annealing = QFormLayout()
        form_pre_annealing.setLayout(layout_pre_annealing)
        layout.addWidget(form_pre_annealing, 2, 2)
        layout_pre_annealing.addRow(
            QLabel("Temperature (\N{DEGREE SIGN}C)"),
            self.pre_annealing_temperature_input[create_index],
        )
        layout_pre_annealing.addRow(
            QLabel("Heating Speed (\N{DEGREE SIGN}C/min)"),
            self.pre_annealing_heating_speed_input[create_index],
        )
        layout_pre_annealing.addRow(
            QLabel("Annealing Time (min)"),
            self.pre_annealing_time_input[create_index],
        )
        layout_pre_annealing.addRow(
            QLabel("Atmosphere Pressure (mTorr)"),
            self.pre_annealing_atmosphere_pressure_input[create_index],
        )

        form_ablation = QGroupBox("Ablation")
        layout_ablation = QFormLayout()
        form_ablation.setLayout(layout_ablation)
        layout.addWidget(form_ablation, 2, 3)
        layout_ablation.addRow(QLabel("Pre-ablation Pulses (count)"), self.pre_number_pulses_input[create_index])
        layout_ablation.addRow(QLabel("Ablation Frequency (Hz)"), self.frequency_input[create_index])
        layout_ablation.addRow(QLabel("Ablation Pulses (count)"), self.number_pulses_input[create_index])

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        return layout

    def add_target(self, set_current: bool = True) -> None:
        """Add one new target page and its corresponding widgets."""
        idx = self._append_target_fields()

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

        while self.Stack.count() > 0:
            page = self.Stack.widget(0)
            self.Stack.removeWidget(page)
            page.deleteLater()

        self.pageCombo.clear()
        self._init_target_storage()

        for _ in range(target_count):
            self.add_target(set_current=False)

        self.pageCombo.setCurrentIndex(0)
        self.switchPage(0)

    def show_message_window(self, message: str) -> None:
        """Display a transient popup with operation status text."""
        self.exPopup = MessageWindow(message)
        self.exPopup.setGeometry(500, 500, 450, 100)
        self.exPopup.show()

    def switchPage(self, index: int) -> None:
        """Switch the active target page in the stacked widget."""
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

    def _apply_info_dict(self, info_dict: Dict[str, Dict[str, Any]]) -> None:
        """Populate form fields from a dictionary loaded from JSON."""
        header = info_dict.get("header", {})

        self._set_line_edit_value(self.growth_id_input, header.get("Growth ID"))
        self._set_line_edit_value(self.name_input, header.get("User Name"))
        self._set_line_edit_value(self.date_input, header.get("Date"))
        self._set_line_edit_value(self.time_input, header.get("time"))
        self._set_line_edit_value(self.save_path_input, header.get("Path"))
        self._set_combo_value(self.chamber_ComboBox, header.get("Chamber"))
        self._set_combo_value(self.cool_down_gas, header.get("Cool Down Atmosphere"))
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
                self.laser_voltage_input[i],
                self._first_present_value(target_dict, "Laser Voltage (kV)", "Laser Voltage(kV)"),
            )
            self._set_line_edit_value(
                self.laser_energy_input[i],
                self._first_present_value(target_dict, "Laser Energy (mJ)", "Laser Energy(mJ)"),
            )
            self._set_line_edit_value(
                self.targeted_measured_energy_input[i],
                self._first_present_value(
                    target_dict,
                    "Targeted Measured Energy (mJ)",
                    "Targeted Measured Energy(mJ)",
                    "Measured Energy Mean(mJ)",
                ),
            )
            self._set_line_edit_value(
                self.fluence_input[i],
                self._first_present_value(target_dict, "Fluence (J/cm^2)", "Fluence"),
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
                self.pre_annealing_temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Temperature (\N{DEGREE SIGN}C)",
                    "Pre-Annealing-Temperature(\N{DEGREE SIGN}C)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_heating_speed_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing-Heating-Speed(\N{DEGREE SIGN}C/min)",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_time_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Time (min)",
                    "Pre-Annealing-Time(min)",
                    "Pre-Annealing-Time",
                ),
            )
            self._set_line_edit_value(
                self.pre_annealing_atmosphere_pressure_input[i],
                self._first_present_value(
                    target_dict,
                    "Pre-Annealing Atmosphere Pressure (mTorr)",
                    "Pre-Annealing-Atmosphere-Pressure(mTorr)",
                ),
            )

            self._set_line_edit_value(
                self.temperature_input[i],
                self._first_present_value(
                    target_dict,
                    "Ablation Temperature (\N{DEGREE SIGN}C)",
                    "Ablation-Temperature(\N{DEGREE SIGN}C)",
                ),
            )
            self._set_line_edit_value(
                self.pressure_input[i],
                self._first_present_value(target_dict, "Ablation Pressure (mTorr)", "Ablation-Pressure(mTorr)"),
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

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                info_dict = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            self.show_message_window(f"Failed to load JSON: {exc}")
            return

        self._apply_info_dict(info_dict)
        self.show_message_window("Parameters loaded!")

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
        for i in range(target_count):
            info_dict[f"target_{i + 1}"] = {
                "Target Material": self.target_input[i].text(),
                "Offset X (mm)": self.offset_x_input[i].text(),
                "Offset Y (mm)": self.offset_y_input[i].text(),
                "Scan Diameter (mm)": self.scan_diameter_input[i].text(),
                "Scan Speed X/Z (mm/s)": self.scan_speed_xz_input[i].text(),
                "Heater Position X (mm)": self.heater_position_x_input[i].text(),
                "Heater Position Y (mm)": self.heater_position_y_input[i].text(),
                "Heater Position Z (mm)": self.heater_position_z_input[i].text(),
                "Tilt (deg)": self.heater_tilt_input[i].text(),
                "Azimuth (deg)": self.heater_azimuth_input[i].text(),
                "Laser Voltage (kV)": self.laser_voltage_input[i].text(),
                "Laser Energy (mJ)": self.laser_energy_input[i].text(),
                "Targeted Measured Energy (mJ)": self.targeted_measured_energy_input[i].text(),
                "Fluence (J/cm^2)": self.fluence_input[i].text(),
                "Mask Width (mm)": self.mask_width_input[i].text(),
                "Mask Height (mm)": self.mask_height_input[i].text(),
                "Mask Area (mm^2)": self.mask_area_input[i].text(),
                "Spot Width (mm)": self.spot_width_input[i].text(),
                "Spot Height (mm)": self.spot_height_input[i].text(),
                "Spot Area (mm^2)": self.spot_area_input[i].text(),
                "Magnification (x)": self.magnification_input[i].text(),
                "Pre-Ablation Pulses (count)": self.pre_number_pulses_input[i].text(),
                "Pre-Annealing Temperature (\N{DEGREE SIGN}C)": self.pre_annealing_temperature_input[i].text(),
                "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)": self.pre_annealing_heating_speed_input[i].text(),
                "Pre-Annealing Time (min)": self.pre_annealing_time_input[i].text(),
                "Pre-Annealing Atmosphere Pressure (mTorr)": self.pre_annealing_atmosphere_pressure_input[
                    i
                ].text(),
                "Ablation Temperature (\N{DEGREE SIGN}C)": self.temperature_input[i].text(),
                "Ablation Pressure (mTorr)": self.pressure_input[i].text(),
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

    @staticmethod
    def _coerce_float(value: Any) -> Any:
        """Return a float when text is numeric; otherwise return original text."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _has_nested_container(value: Any) -> bool:
        """Return True when dict/list contains nested dict/list values."""
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, (dict, list)) or GenerateForm._has_nested_container(item):
                    return True
            return False
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)) or GenerateForm._has_nested_container(item):
                    return True
            return False
        return False

    @staticmethod
    def _flatten_rows(value: Any, prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested dict/list data into parameter/value rows."""
        rows: List[Dict[str, Any]] = []
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(GenerateForm._flatten_rows(item, key_path))
            return rows

        if isinstance(value, list):
            if not value:
                rows.append({"parameter": prefix, "value": ""})
                return rows
            for index, item in enumerate(value):
                key_path = f"{prefix}[{index}]"
                rows.extend(GenerateForm._flatten_rows(item, key_path))
            return rows

        rows.append({"parameter": prefix, "value": value})
        return rows

    @staticmethod
    def _to_table_rows(data: Any):
        """Convert input data into simple headers/rows for table rendering."""
        if isinstance(data, dict):
            if GenerateForm._has_nested_container(data):
                flat = GenerateForm._flatten_rows(data)
                return ["parameter", "value"], [[r["parameter"], r["value"]] for r in flat]
            return ["parameter", "value"], [[k, v] for k, v in data.items()]

        if isinstance(data, list):
            if data and all(isinstance(item, dict) and not GenerateForm._has_nested_container(item) for item in data):
                headers: List[str] = []
                for item in data:
                    for key in item.keys():
                        if key not in headers:
                            headers.append(key)
                rows = [[item.get(h, "") for h in headers] for item in data]
                return headers, rows

            if any(isinstance(item, (dict, list)) for item in data):
                flat = GenerateForm._flatten_rows(data)
                return ["parameter", "value"], [[r["parameter"], r["value"]] for r in flat]

            return ["value"], [[item] for item in data]

        return ["value"], [[data]]

    @staticmethod
    def _render_html_table(headers: List[str], rows: List[List[Any]], show_header: bool = False) -> str:
        """Build HTML markup for one data table."""
        header_cells = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
        row_markup: List[str] = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
            row_markup.append(f"<tr>{cells}</tr>")
        colspan = max(1, len(headers))
        body = "".join(row_markup) if row_markup else f"<tr><td colspan='{colspan}'>&nbsp;</td></tr>"
        if show_header:
            return f"<table><thead><tr>{header_cells}</tr></thead><tbody>{body}</tbody></table>"
        return f"<table><tbody>{body}</tbody></table>"

    @staticmethod
    def _target_section_spec() -> List[tuple[str, List[str]]]:
        """Return ordered target section definitions used in HTML export."""
        return [
            (
                "Target Parameters",
                [
                    "Target Material",
                    "Offset X (mm)",
                    "Offset Y (mm)",
                    "Scan Diameter (mm)",
                    "Scan Speed X/Z (mm/s)",
                ],
            ),
            (
                "Heater Parameter",
                [
                    "Heater Position X (mm)",
                    "Heater Position Y (mm)",
                    "Heater Position Z (mm)",
                    "Tilt (deg)",
                    "Azimuth (deg)",
                ],
            ),
            (
                "Laser and Mask",
                [
                    "Laser Voltage (kV)",
                    "Laser Energy (mJ)",
                    "Targeted Measured Energy (mJ)",
                    "Fluence (J/cm^2)",
                ],
            ),
            (
                "Mask and Spot",
                [
                    "Mask Width (mm)",
                    "Mask Height (mm)",
                    "Mask Area (mm^2)",
                    "Spot Width (mm)",
                    "Spot Height (mm)",
                    "Spot Area (mm^2)",
                    "Magnification (x)",
                ],
            ),
            (
                "Pre-annealing",
                [
                    "Pre-Annealing Temperature (\N{DEGREE SIGN}C)",
                    "Pre-Annealing Heating Speed (\N{DEGREE SIGN}C/min)",
                    "Pre-Annealing Time (min)",
                    "Pre-Annealing Atmosphere Pressure (mTorr)",
                ],
            ),
            (
                "Ablation",
                [
                    "Pre-Ablation Pulses (count)",
                    "Ablation Temperature (\N{DEGREE SIGN}C)",
                    "Ablation Pressure (mTorr)",
                    "Ablation Atmosphere Gas",
                    "Ablation Frequency (Hz)",
                    "Ablation Pulses (count)",
                ],
            ),
        ]

    @staticmethod
    def _render_target_sections(target_data: Dict[str, Any]) -> str:
        """Render grouped target tables by form section for HTML output."""
        sections: List[str] = []
        used_keys: set[str] = set()

        for section_title, section_keys in GenerateForm._target_section_spec():
            rows = [[key, target_data[key]] for key in section_keys if key in target_data]
            if not rows:
                continue
            used_keys.update(key for key in section_keys if key in target_data)
            table_markup = GenerateForm._render_html_table(["parameter", "value"], rows)
            sections.append(
                "<section class='subcard'>"
                f"<h3>{html.escape(section_title)}</h3>"
                f"{table_markup}"
                "</section>"
            )

        remaining_rows = [[key, value] for key, value in target_data.items() if key not in used_keys]
        if remaining_rows:
            table_markup = GenerateForm._render_html_table(["parameter", "value"], remaining_rows)
            sections.append(
                "<section class='subcard'>"
                "<h3>Other</h3>"
                f"{table_markup}"
                "</section>"
            )

        return "".join(sections)

    @staticmethod
    def _write_html(file_path: str, data: Any) -> None:
        """Write form data to a styled HTML report for OneNote import."""
        cards: List[str] = []
        if isinstance(data, dict):
            header_data = data.get("header")
            if isinstance(header_data, dict):
                headers, rows = GenerateForm._to_table_rows(header_data)
                cards.append(
                    "<section class='card'>"
                    "<h2>Header</h2>"
                    f"{GenerateForm._render_html_table(headers, rows)}"
                    "</section>"
                )

            target_keys = [key for key in data.keys() if key.lower().startswith("target_")]
            target_keys.sort(key=lambda key: int(key.split("_")[1]) if key.split("_")[1].isdigit() else key)
            for target_key in target_keys:
                target_data = data.get(target_key)
                if not isinstance(target_data, dict):
                    headers, rows = GenerateForm._to_table_rows(target_data)
                    target_markup = GenerateForm._render_html_table(headers, rows)
                else:
                    target_markup = f"<div class='section-grid'>{GenerateForm._render_target_sections(target_data)}</div>"

                cards.append(
                    "<section class='card'>"
                    f"<h2>{html.escape(target_key.replace('_', ' ').title())}</h2>"
                    f"{target_markup}"
                    "</section>"
                )

            other_sections = [
                (key, value)
                for key, value in data.items()
                if key != "header" and key not in target_keys
            ]
            for section_name, section_data in other_sections:
                headers, rows = GenerateForm._to_table_rows(section_data)
                cards.append(
                    "<section class='card'>"
                    f"<h2>{html.escape(str(section_name))}</h2>"
                    f"{GenerateForm._render_html_table(headers, rows)}"
                    "</section>"
                )
        else:
            headers, rows = GenerateForm._to_table_rows(data)
            cards.append(
                "<section class='card'>"
                "<h2>Parameters</h2>"
                f"{GenerateForm._render_html_table(headers, rows)}"
                "</section>"
            )

        page_title = "PLD Growth Parameters"
        now_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --line: #d8e1ec;
      --text: #182230;
      --muted: #526178;
      --head: #ebf2f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Calibri, Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    .meta {{
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
      color: #243955;
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .subcard {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfdff;
    }}
    .subcard h3 {{
      margin: 0 0 8px 0;
      font-size: 14px;
      color: #2d4b72;
      letter-spacing: 0.2px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: var(--head);
      text-transform: capitalize;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(page_title)}</h1>
  <div class="meta">Exported: {html.escape(now_stamp)}</div>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(document)

    def _default_file_stem(self) -> str:
        """Build output file stem from core header fields."""
        growth_id = self.growth_id_input.text().strip()
        user_name = self.name_input.text().strip()
        date_stamp = "".join(self.date_input.text().split("/")).strip()

        if growth_id or user_name:
            return f"{growth_id}_{user_name}_{date_stamp}".strip("_")

        if not date_stamp:
            date_stamp = datetime.datetime.today().strftime("%m%d%Y")
        return f"growth_record_{date_stamp}"

    def save(self) -> None:
        """Serialize the current form state to JSON and HTML files."""
        print("Saving dictionary...")

        output_dir = self.save_path_input.text().strip() or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        self.file_name = self._default_file_stem()
        self.path = output_dir

        self.info_dict = self.get_info()
        for section in self.info_dict.values():
            for key, value in list(section.items()):
                section[key] = self._coerce_float(value)

        output_file = os.path.join(output_dir, f"{self.file_name}.json")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(self.info_dict, file)

        html_file = os.path.join(output_dir, f"{self.file_name}.html")
        try:
            self._write_html(html_file, self.info_dict)
            print(f"Done! Saved: {output_file}")
            print(f"Done! Saved: {html_file}")
            self.show_message_window("Parameters saved to JSON and HTML!")
        except Exception as exc:
            print(f"Done! Saved: {output_file}")
            print(f"HTML export failed: {exc}")
            self.show_message_window(f"JSON saved. HTML export failed: {exc}")

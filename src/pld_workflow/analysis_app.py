"""Optional desktop helper for exploring parameter trends across JSON records."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .analysis import build_parameter_trend, list_available_parameters, plot_parameter_trend


class ParameterTrendWindow(QWidget):
    """Small analysis window kept separate from the everyday recorder workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD Parameter Trend Analyzer")
        self.setMinimumSize(920, 680)

        self._analysis_sources: list[str] = []

        self.source_input = QLineEdit()
        self.parameter_combo = QComboBox()
        self.parameter_combo.setEditable(True)
        self.section_combo = QComboBox()
        self.section_combo.setEditable(True)
        self.summary_preview = QPlainTextEdit()
        self.summary_preview.setReadOnly(True)
        self.status_label = QLabel(
            "Choose a JSON record directory or a set of JSON files, scan available parameters, then plot one trend."
        )
        self.status_label.setWordWrap(True)

        self._init_layout()

    def _init_layout(self) -> None:
        """Build the analyzer layout."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        source_group = QGroupBox("Record Source")
        source_form = QFormLayout()
        source_group.setLayout(source_form)
        main_layout.addWidget(source_group)

        source_row = QHBoxLayout()
        browse_dir_button = QPushButton("Browse Directory")
        browse_dir_button.clicked.connect(self._choose_directory)
        browse_files_button = QPushButton("Browse Files")
        browse_files_button.clicked.connect(self._choose_files)
        source_row.addWidget(self.source_input)
        source_row.addWidget(browse_dir_button)
        source_row.addWidget(browse_files_button)
        source_form.addRow(QLabel("JSON Records"), _wrap_layout(source_row))

        options_group = QGroupBox("Trend Options")
        options_form = QFormLayout()
        options_group.setLayout(options_form)
        main_layout.addWidget(options_group)
        options_form.addRow(QLabel("Parameter"), self.parameter_combo)
        options_form.addRow(QLabel("Section"), self.section_combo)

        action_row = QHBoxLayout()
        scan_button = QPushButton("Scan Records")
        scan_button.clicked.connect(self._scan_records)
        plot_button = QPushButton("Plot Trend")
        plot_button.clicked.connect(self._plot_trend)
        action_row.addWidget(scan_button)
        action_row.addWidget(plot_button)
        main_layout.addLayout(action_row)

        summary_group = QGroupBox("Available Parameters")
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        summary_layout.addWidget(self.summary_preview)
        main_layout.addWidget(summary_group, 1)
        main_layout.addWidget(self.status_label)

    def _choose_directory(self) -> None:
        """Select one directory containing many JSON records."""
        selected = QFileDialog.getExistingDirectory(self, "Select JSON Record Directory")
        if not selected:
            return
        self._analysis_sources = [selected]
        self.source_input.setText(selected)
        self._set_status(f"Selected record directory: {selected}")

    def _choose_files(self) -> None:
        """Select a list of individual JSON files for analysis."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select JSON Records", "", "JSON Files (*.json)")
        if not files:
            return
        self._analysis_sources = files
        self.source_input.setText("; ".join(files))
        self._set_status(f"Selected {len(files)} JSON files.")

    def _scan_records(self) -> None:
        """Scan the chosen records and refresh the parameter/section lists."""
        if not self._analysis_sources:
            self._analysis_sources = _sources_from_text(self.source_input.text())
        if not self._analysis_sources:
            self._set_status("Choose a directory or JSON files first.")
            return

        try:
            summary = list_available_parameters(self._analysis_sources)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Scan failed: {exc}")
            return

        self.parameter_combo.clear()
        self.section_combo.clear()
        self.section_combo.addItem("")
        self.summary_preview.clear()

        if summary.empty:
            self._set_status("No JSON records were found in the selected source.")
            return

        for parameter in summary["parameter"].dropna().drop_duplicates().tolist():
            self.parameter_combo.addItem(str(parameter))
        for section in summary["section"].dropna().drop_duplicates().tolist():
            self.section_combo.addItem(str(section))

        self.summary_preview.setPlainText(summary.to_string(index=False))
        self._set_status(f"Scanned {len(summary)} section/parameter combinations.")

    def _plot_trend(self) -> None:
        """Build and show a plot for the requested parameter trend."""
        if not self._analysis_sources:
            self._analysis_sources = _sources_from_text(self.source_input.text())
        if not self._analysis_sources:
            self._set_status("Choose a directory or JSON files first.")
            return

        parameter = self.parameter_combo.currentText().strip()
        section = self.section_combo.currentText().strip() or None
        if not parameter:
            self._set_status("Choose or type a parameter name first.")
            return

        try:
            trend = build_parameter_trend(self._analysis_sources, parameter=parameter, section=section)
            ax = plot_parameter_trend(trend)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Plot failed: {exc}")
            return

        ax.figure.show()
        self._set_status(f"Plotted trend for '{parameter}'.")

    def _set_status(self, message: str) -> None:
        """Update the one-line status message."""
        self.status_label.setText(message)


def _sources_from_text(text: str) -> list[str]:
    """Split a semicolon-separated source list from the line edit."""
    return [item.strip() for item in text.split(";") if item.strip()]


def _wrap_layout(layout) -> QWidget:
    """Wrap a layout for use inside form rows."""
    container = QWidget()
    container.setLayout(layout)
    return container


def main() -> int:
    """Launch the optional parameter trend analyzer window."""
    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)
    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QLabel")
    app.setFont(custom_font, "QLineEdit")
    app.setFont(custom_font, "QPlainTextEdit")

    window = ParameterTrendWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

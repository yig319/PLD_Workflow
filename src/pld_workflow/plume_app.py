"""Standalone plume-management application separated from the PLD recorder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .plume_management import (
    PlumePackResult,
    build_plume_archive_stem,
    metadata_to_text,
    pack_plume_directory,
    read_metadata_json,
    upload_archive_to_datafed,
    write_metadata_json,
)


class PlumeManagerWindow(QWidget):
    """Desktop UI for packaging plume image folders and editing linked metadata."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD Plume Manager")
        self.setMinimumSize(980, 760)

        self._last_dir = os.getcwd()
        self._metadata_file_path = ""

        self._init_inputs()
        self._init_layout()
        self._update_archive_name_from_inputs()

    def _init_inputs(self) -> None:
        """Create the editable widgets used by the plume manager."""
        self.source_dir_input = QLineEdit(os.getcwd())
        self.output_dir_input = QLineEdit(os.getcwd())
        self.archive_name_input = QLineEdit()
        self.metadata_file_input = QLineEdit()
        self.dataset_id_input = QLineEdit("c/391937642")
        self.source_dir_input.textChanged.connect(self._update_archive_name_from_inputs)

        self.metadata_editor = QPlainTextEdit("{}")
        self.metadata_editor.setPlaceholderText(
            '{\n  "header": {\n    "Growth ID": "example"\n  }\n}'
        )
        self.metadata_editor.textChanged.connect(self._update_archive_name_from_inputs)

        self.result_preview = QPlainTextEdit()
        self.result_preview.setReadOnly(True)

        self.status_label = QLabel(
            "Select a plume directory, optionally load a parameter JSON record, then pack to HDF5."
        )
        self.status_label.setWordWrap(True)

    def _init_layout(self) -> None:
        """Build the window layout."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        path_layout = QGridLayout()
        main_layout.addLayout(path_layout)

        source_group = QGroupBox("Plume Dataset")
        source_form = QFormLayout()
        source_group.setLayout(source_form)
        path_layout.addWidget(source_group, 0, 0)

        source_row = QHBoxLayout()
        browse_source_button = QPushButton("Browse...")
        browse_source_button.clicked.connect(self._choose_source_dir)
        source_row.addWidget(self.source_dir_input)
        source_row.addWidget(browse_source_button)
        source_form.addRow(QLabel("Source Directory"), _wrap_layout(source_row))

        output_group = QGroupBox("Archive Output")
        output_form = QFormLayout()
        output_group.setLayout(output_form)
        path_layout.addWidget(output_group, 0, 1)

        output_dir_row = QHBoxLayout()
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self._choose_output_dir)
        output_dir_row.addWidget(self.output_dir_input)
        output_dir_row.addWidget(browse_output_button)
        output_form.addRow(QLabel("Output Directory"), _wrap_layout(output_dir_row))
        output_form.addRow(QLabel("Archive Name"), self.archive_name_input)
        output_form.addRow(QLabel("DataFed Collection"), self.dataset_id_input)

        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout()
        metadata_group.setLayout(metadata_layout)
        main_layout.addWidget(metadata_group, 1)

        metadata_path_form = QFormLayout()
        metadata_path_row = QHBoxLayout()
        load_metadata_button = QPushButton("Load JSON")
        load_metadata_button.clicked.connect(self._load_metadata_from_dialog)
        save_metadata_button = QPushButton("Save JSON")
        save_metadata_button.clicked.connect(self._save_metadata_to_dialog)
        metadata_path_row.addWidget(self.metadata_file_input)
        metadata_path_row.addWidget(load_metadata_button)
        metadata_path_row.addWidget(save_metadata_button)
        metadata_path_form.addRow(QLabel("Metadata File"), _wrap_layout(metadata_path_row))
        metadata_layout.addLayout(metadata_path_form)
        metadata_layout.addWidget(self.metadata_editor, 1)

        action_row = QHBoxLayout()
        pack_button = QPushButton("Pack to HDF5")
        pack_button.clicked.connect(self._pack_archive)
        upload_button = QPushButton("Pack and Upload")
        upload_button.clicked.connect(self._pack_and_upload)
        action_row.addWidget(pack_button)
        action_row.addWidget(upload_button)
        main_layout.addLayout(action_row)

        results_group = QGroupBox("Result Summary")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        results_layout.addWidget(self.result_preview)
        main_layout.addWidget(results_group, 1)
        main_layout.addWidget(self.status_label)

    def _choose_source_dir(self) -> None:
        """Choose the plume dataset directory."""
        start_dir = self.source_dir_input.text().strip() or self._last_dir
        selected = QFileDialog.getExistingDirectory(self, "Select Plume Source Directory", start_dir)
        if not selected:
            return
        self._last_dir = selected
        self.source_dir_input.setText(selected)
        self._update_archive_name_from_inputs()
        self._set_status(f"Selected plume source: {selected}")

    def _choose_output_dir(self) -> None:
        """Choose the directory where the packed HDF5 file should be written."""
        start_dir = self.output_dir_input.text().strip() or self._last_dir
        selected = QFileDialog.getExistingDirectory(self, "Select Archive Output Directory", start_dir)
        if not selected:
            return
        self._last_dir = selected
        self.output_dir_input.setText(selected)
        self._set_status(f"Selected output directory: {selected}")

    def _load_metadata_from_dialog(self) -> None:
        """Load a PLD parameter JSON file into the editable metadata panel."""
        start_dir = self.source_dir_input.text().strip() or self._last_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Metadata JSON",
            start_dir,
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            metadata = read_metadata_json(file_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to load metadata JSON: {exc}")
            return

        self._metadata_file_path = file_path
        self.metadata_file_input.setText(file_path)
        self.metadata_editor.setPlainText(metadata_to_text(metadata))
        self._set_status(f"Loaded metadata JSON: {file_path}")

    def _save_metadata_to_dialog(self) -> None:
        """Save the edited metadata JSON to disk."""
        try:
            metadata = self._current_metadata()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot save metadata: {exc}")
            return
        start_dir = self.output_dir_input.text().strip() or self._last_dir
        default_name = Path(self._metadata_file_path).name if self._metadata_file_path else "pld_metadata.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metadata JSON",
            str(Path(start_dir) / default_name),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            write_metadata_json(metadata, file_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to save metadata JSON: {exc}")
            return

        self._metadata_file_path = file_path
        self.metadata_file_input.setText(file_path)
        self._set_status(f"Saved metadata JSON: {file_path}")

    def _pack_archive(self) -> None:
        """Create the HDF5 plume archive and update the on-screen summary."""
        try:
            output_path = self._archive_output_path()
            metadata = self._optional_metadata()
            result = pack_plume_directory(self.source_dir_input.text(), output_path, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Pack failed: {exc}")
            return

        self.result_preview.setPlainText(self._format_pack_result(result))
        self._set_status(f"Packed plume archive: {result.output_path}")

    def _pack_and_upload(self) -> None:
        """Create the archive first, then upload it to DataFed."""
        try:
            metadata = self._current_metadata()
            result = pack_plume_directory(
                self.source_dir_input.text(),
                self._archive_output_path(),
                metadata=metadata,
            )
            upload_result = upload_archive_to_datafed(
                result.output_path,
                metadata,
                collection_id=self.dataset_id_input.text().strip() or "c/391937642",
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Pack/upload failed: {exc}")
            return

        summary = self._format_pack_result(result)
        summary += (
            "\n\nDataFed Upload\n"
            f"record_id: {upload_result['record_id']}\n"
            "transfer: completed"
        )
        self.result_preview.setPlainText(summary)
        self._set_status(f"Uploaded archive to DataFed record {upload_result['record_id']}")

    def _archive_output_path(self) -> str:
        """Return the full HDF5 output path derived from the current form fields."""
        source_dir = self.source_dir_input.text().strip()
        if not source_dir:
            raise ValueError("Choose a plume source directory first.")

        output_dir = self.output_dir_input.text().strip() or source_dir
        archive_name = self.archive_name_input.text().strip()
        if not archive_name:
            self._update_archive_name_from_inputs()
            archive_name = self.archive_name_input.text().strip()
        if not archive_name:
            raise ValueError("Provide an archive name before packing.")

        return str(Path(output_dir).expanduser().resolve() / f"{archive_name}.h5")

    def _optional_metadata(self) -> dict[str, Any] | None:
        """Return metadata when the editor contains JSON, otherwise None."""
        text = self.metadata_editor.toPlainText().strip()
        if not text:
            return None
        return self._current_metadata()

    def _current_metadata(self) -> dict[str, Any]:
        """Parse and return the JSON currently shown in the metadata editor."""
        text = self.metadata_editor.toPlainText().strip() or "{}"
        try:
            metadata = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Metadata JSON is invalid: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")
        return metadata

    def _update_archive_name_from_inputs(self) -> None:
        """Refresh the suggested archive name from the selected source and metadata."""
        source_dir = self.source_dir_input.text().strip()
        if not source_dir:
            return
        try:
            metadata = self._optional_metadata()
        except ValueError:
            metadata = None
        suggested = build_plume_archive_stem(source_dir, metadata=metadata)
        current = self.archive_name_input.text().strip()
        if not current or current.endswith("_plume"):
            self.archive_name_input.setText(suggested)

    def _format_pack_result(self, result: PlumePackResult) -> str:
        """Format a readable multi-line summary for the result panel."""
        lines = [
            f"output_path: {result.output_path}",
            f"targets: {result.total_targets}",
            f"plume_folders: {result.total_plumes}",
            f"frames: {result.total_frames}",
            f"desktop_ini_removed: {result.removed_ini_files}",
            "",
            "Target Details:",
        ]
        for item in result.target_summaries:
            lines.append(
                f"- {item.target_name}: {item.plume_count} plume folders, "
                f"{item.frame_count} frames, frame shape {item.frame_shape}"
            )
        return "\n".join(lines)

    def _set_status(self, message: str) -> None:
        """Update the status text shown at the bottom of the window."""
        self.status_label.setText(message)


def _wrap_layout(layout) -> QWidget:
    """Wrap a layout in a QWidget so it can be placed in a QFormLayout row."""
    container = QWidget()
    container.setLayout(layout)
    return container


def main() -> int:
    """Launch the standalone plume-management desktop application."""
    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)
    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QLabel")
    app.setFont(custom_font, "QLineEdit")
    app.setFont(custom_font, "QPlainTextEdit")

    window = PlumeManagerWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

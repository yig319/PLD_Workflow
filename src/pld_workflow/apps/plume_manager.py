"""Standalone plume-management application separated from the PLD recorder."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..plume_management import (
    PlumeArchiveRecord,
    PlumeDatasetRecord,
    PlumePackResult,
    PlumeWorkspaceCreationResult,
    RawFileStagingResult,
    build_plume_growth_stem,
    build_plume_workspace_targets,
    create_plume_workspace,
    inspect_plume_archive,
    metadata_to_text,
    pack_plume_directory,
    read_metadata_json,
    read_packed_frame,
    read_plume_frame,
    scan_plume_directory,
    stage_raw_files_for_target,
    write_metadata_json,
)


ROLE_PATH = Qt.UserRole
ROLE_KIND = Qt.UserRole + 1
ROLE_TARGET_ROOT = Qt.UserRole + 2
ROLE_ARCHIVE_TARGET = Qt.UserRole + 3
ROLE_PLUME_INDEX = Qt.UserRole + 4
ROLE_FRAME_INDEX = Qt.UserRole + 5
ROLE_FRAME_COUNT = Qt.UserRole + 6

IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class PlumeManagerWindow(QWidget):
    """Desktop UI for plume workspace creation, raw staging, packing, and HDF5 review."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD Plume Manager")
        self.setMinimumSize(1320, 860)

        self._last_dir = os.getcwd()
        self._metadata_file_path = ""
        self._workspace_record: PlumeDatasetRecord | None = None
        self._archive_record: PlumeArchiveRecord | None = None
        self._preview_pixmap: QPixmap | None = None
        self._video_frames: list[dict[str, Any]] = []
        self._last_suggested_growth_dir = ""

        self._init_inputs()
        self._init_layout()
        self._update_growth_folder_from_inputs()
        self._refresh_metadata_summary()
        self._refresh_archive_path_preview()
        self._reload_raw_inbox()
        self._reload_workspace_tree()
        self._refresh_action_buttons()

    def _init_inputs(self) -> None:
        """Create the editable widgets used by the plume manager."""

        self.storage_root_input = QLineEdit()
        self.storage_root_input.setPlaceholderText(
            "Choose the root folder that stores the JSON file, recorder date folders, and created growth folders."
        )
        self.workspace_root_input = QLineEdit()
        self.workspace_root_input.setReadOnly(True)
        self.workspace_root_input.setPlaceholderText(
            "This growth folder is suggested from the loaded JSON, or you can browse an existing one."
        )
        self.raw_inbox_input = QLineEdit()
        self.raw_inbox_input.setPlaceholderText(
            "Choose the recorder date folder that currently contains new raw files waiting to be moved."
        )
        self.archive_file_input = QLineEdit()
        self.archive_file_input.setPlaceholderText("Browse to an existing H5 file if you want to inspect a packed archive.")
        self.metadata_file_input = QLineEdit()
        self.metadata_file_input.setPlaceholderText("Load the PLD JSON file stored in the root folder.")

        self.storage_root_input.textChanged.connect(self._handle_storage_root_changed)
        self.workspace_root_input.textChanged.connect(self._handle_workspace_root_changed)

        self.metadata_editor = QPlainTextEdit("")
        self.metadata_editor.textChanged.connect(self._handle_metadata_changed)

        self.workflow_help_label = QLabel(
            "<b>Simple workflow:</b> 1) Load the JSON in the root folder and create the growth folder. "
            "This creates folders like <code>1-target</code> and <code>1-target-Pre</code>. "
            "2) Move raw files from the recorder date folder into the correct target folder. "
            "3) Decode with the external program so it creates <code>BMP</code>, then pack to H5 or open an existing H5 for viewing."
        )
        self.workflow_help_label.setWordWrap(True)
        self.workflow_help_label.setTextFormat(Qt.RichText)

        self.path_help_label = QLabel(
            "The storage root is the main working folder for this growth. "
            "Loading the JSON can set this automatically from the JSON file location."
        )
        self.path_help_label.setWordWrap(True)

        self.raw_help_label = QLabel(
            "Use the latest recorder date folder or browse one manually, then move selected raw files directly into the destination target folder."
        )
        self.raw_help_label.setWordWrap(True)

        self.archive_help_label = QLabel(
            "Packing uses the growth folder and saves an H5 with the same base name back into the storage root. "
            "Decoded images can be in <code>target/BMP/frame.png</code> or <code>target/BMP/plume_xxx/frame.png</code>."
        )
        self.archive_help_label.setWordWrap(True)
        self.archive_help_label.setTextFormat(Qt.RichText)

        self.metadata_summary_label = QLabel(
            "Load the JSON file first. The growth folder name and target folders will be previewed here."
        )
        self.metadata_summary_label.setWordWrap(True)

        self.archive_path_preview_label = QLabel("Packed H5 path will appear here after a growth folder is chosen.")
        self.archive_path_preview_label.setWordWrap(True)

        self.raw_target_label = QLabel("Destination target: none")
        self.raw_target_label.setWordWrap(True)
        self.target_selector = QComboBox()
        self.target_selector.currentIndexChanged.connect(self._handle_target_selector_changed)

        self.raw_inbox_list = QListWidget()
        self.raw_inbox_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.workspace_summary_label = QLabel("Choose or create a growth folder to browse its target structure.")
        self.workspace_summary_label.setWordWrap(True)

        self.workspace_tree = QTreeWidget()
        self.workspace_tree.setColumnCount(3)
        self.workspace_tree.setHeaderLabels(["Name", "Kind", "Details"])
        self.workspace_tree.itemSelectionChanged.connect(self._handle_workspace_selection)

        self.archive_summary_label = QLabel("Load or build an H5 archive to inspect its packed plume tree.")
        self.archive_summary_label.setWordWrap(True)

        self.archive_tree = QTreeWidget()
        self.archive_tree.setColumnCount(3)
        self.archive_tree.setHeaderLabels(["Name", "Kind", "Details"])
        self.archive_tree.itemSelectionChanged.connect(self._handle_archive_selection)

        self.preview_image_label = QLabel("Select a plume folder, BMP folder, or H5 plume to preview it here.")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setMinimumHeight(360)
        self.preview_image_label.setStyleSheet("background: #f8fafc; border: 1px solid #d7e0ea;")

        self.preview_info_label = QLabel("Preview details will appear here.")
        self.preview_info_label.setWordWrap(True)

        self.previous_frame_button = QPushButton("Previous")
        self.previous_frame_button.clicked.connect(lambda: self._step_video_frame(-1))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._handle_frame_slider_changed)
        self.next_frame_button = QPushButton("Next")
        self.next_frame_button.clicked.connect(lambda: self._step_video_frame(1))
        self.frame_position_label = QLabel("Frame 0 / 0")
        self.frame_position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.status_label = QLabel(
            "Load the JSON file, create the growth folder, move raw files into the correct target, then decode and pack to H5."
        )
        self.status_label.setWordWrap(True)

    def _init_layout(self) -> None:
        """Build the window layout."""

        self.setStyleSheet(
            """
            QWidget {
                background: #f3f6f8;
                color: #172331;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d3dde6;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #25445e;
                background: #f3f6f8;
            }
            QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTreeWidget {
                background: #ffffff;
                border: 1px solid #c7d3df;
                border-radius: 6px;
                padding: 4px 6px;
            }
            QPushButton {
                background: #e8eef4;
                border: 1px solid #c5d0dc;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #dde8f1;
            }
            QPushButton:disabled {
                color: #8a99a8;
                background: #edf1f5;
            }
            QTabWidget::pane {
                border: 1px solid #d3dde6;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e7edf3;
                border: 1px solid #cbd6e1;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #17384f;
            }
            """
        )

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        workflow_group = QGroupBox("Workflow")
        workflow_layout = QVBoxLayout()
        workflow_group.setLayout(workflow_layout)
        workflow_layout.addWidget(self.workflow_help_label)
        main_layout.addWidget(workflow_group)

        top_grid = QGridLayout()
        top_grid.setColumnStretch(0, 3)
        top_grid.setColumnStretch(1, 3)
        top_grid.setColumnStretch(2, 1)
        main_layout.addLayout(top_grid)

        workspace_group = QGroupBox("Step 1. Load JSON and Create Growth Folder")
        workspace_form = QFormLayout()
        workspace_group.setLayout(workspace_form)
        top_grid.addWidget(workspace_group, 0, 0)
        workspace_form.addRow(self.path_help_label)

        storage_root_row = QHBoxLayout()
        browse_storage_button = QPushButton("Browse...")
        browse_storage_button.clicked.connect(lambda: self._choose_directory_for_input(self.storage_root_input))
        storage_root_row.addWidget(self.storage_root_input)
        storage_root_row.addWidget(browse_storage_button)
        workspace_form.addRow(QLabel("Storage Root"), _wrap_layout(storage_root_row))

        metadata_row = QHBoxLayout()
        load_metadata_button = QPushButton("Load JSON")
        load_metadata_button.clicked.connect(self._load_metadata_from_dialog)
        metadata_row.addWidget(self.metadata_file_input)
        metadata_row.addWidget(load_metadata_button)
        workspace_form.addRow(QLabel("Metadata JSON"), _wrap_layout(metadata_row))

        workspace_root_row = QHBoxLayout()
        browse_workspace_button = QPushButton("Open Existing...")
        browse_workspace_button.clicked.connect(lambda: self._choose_directory_for_input(self.workspace_root_input))
        refresh_workspace_button = QPushButton("Refresh View")
        refresh_workspace_button.clicked.connect(self._reload_workspace_tree)
        workspace_root_row.addWidget(self.workspace_root_input)
        workspace_root_row.addWidget(browse_workspace_button)
        workspace_root_row.addWidget(refresh_workspace_button)
        workspace_form.addRow(QLabel("Growth Folder"), _wrap_layout(workspace_root_row))
        workspace_form.addRow(self.metadata_summary_label)

        self.create_workspace_button = QPushButton("Create Growth Folder from JSON")
        self.create_workspace_button.clicked.connect(self._create_workspace_from_metadata)
        workspace_form.addRow(self.create_workspace_button)

        raw_group = QGroupBox("Step 2. Move Raw Data into the Correct Target")
        raw_layout = QVBoxLayout()
        raw_group.setLayout(raw_layout)
        top_grid.addWidget(raw_group, 0, 1)
        raw_layout.addWidget(self.raw_target_label)
        raw_layout.addWidget(self.raw_help_label)
        raw_target_row = QHBoxLayout()
        raw_target_row.addWidget(QLabel("Destination Target"))
        raw_target_row.addWidget(self.target_selector, 1)
        raw_layout.addWidget(_wrap_layout(raw_target_row))
        raw_inbox_row = QHBoxLayout()
        browse_raw_button = QPushButton("Browse...")
        browse_raw_button.clicked.connect(lambda: self._choose_directory_for_input(self.raw_inbox_input))
        latest_raw_button = QPushButton("Use Latest Date Folder")
        latest_raw_button.clicked.connect(self._set_raw_inbox_to_latest_date_folder)
        refresh_raw_button = QPushButton("Refresh")
        refresh_raw_button.clicked.connect(self._reload_raw_inbox)
        raw_inbox_row.addWidget(self.raw_inbox_input)
        raw_inbox_row.addWidget(browse_raw_button)
        raw_inbox_row.addWidget(latest_raw_button)
        raw_inbox_row.addWidget(refresh_raw_button)
        raw_layout.addWidget(QLabel("Recorder Date Folder"))
        raw_layout.addWidget(_wrap_layout(raw_inbox_row))
        raw_layout.addWidget(self.raw_inbox_list, 1)
        self.move_raw_button = QPushButton("Move Selected Raw Files to Destination Target")
        self.move_raw_button.clicked.connect(self._move_selected_raw_files)
        self.move_all_raw_button = QPushButton("Move All Raw Files to Destination Target")
        self.move_all_raw_button.clicked.connect(self._move_all_raw_files)
        raw_move_row = QHBoxLayout()
        raw_move_row.addWidget(self.move_raw_button)
        raw_move_row.addWidget(self.move_all_raw_button)
        raw_layout.addWidget(_wrap_layout(raw_move_row))

        archive_group = QGroupBox("Step 3. Pack to H5 or Open an Existing H5")
        archive_form = QFormLayout()
        archive_group.setLayout(archive_form)
        top_grid.addWidget(archive_group, 0, 2)
        archive_form.addRow(self.archive_help_label)
        archive_form.addRow(QLabel("Packed H5 Path"), self.archive_path_preview_label)

        self.pack_button = QPushButton("Pack Growth Folder to HDF5")
        self.pack_button.clicked.connect(self._pack_archive)
        archive_form.addRow(self.pack_button)

        archive_file_row = QHBoxLayout()
        browse_archive_button = QPushButton("Browse...")
        browse_archive_button.clicked.connect(self._choose_archive_file)
        load_archive_button = QPushButton("Load H5")
        load_archive_button.clicked.connect(self._load_archive_tree)
        archive_file_row.addWidget(self.archive_file_input)
        archive_file_row.addWidget(browse_archive_button)
        archive_file_row.addWidget(load_archive_button)
        archive_form.addRow(QLabel("Existing H5"), _wrap_layout(archive_file_row))

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        browser_panel = QWidget()
        browser_layout = QVBoxLayout()
        browser_panel.setLayout(browser_layout)
        self.browser_tabs = QTabWidget()
        browser_layout.addWidget(self.browser_tabs)

        workspace_tab = QWidget()
        workspace_tab_layout = QVBoxLayout()
        workspace_tab.setLayout(workspace_tab_layout)
        workspace_tab_layout.addWidget(self.workspace_summary_label)
        workspace_tab_layout.addWidget(self.workspace_tree, 1)
        self.browser_tabs.addTab(workspace_tab, "Growth Folder")

        archive_tab = QWidget()
        archive_tab_layout = QVBoxLayout()
        archive_tab.setLayout(archive_tab_layout)
        archive_tab_layout.addWidget(self.archive_summary_label)
        archive_tab_layout.addWidget(self.archive_tree, 1)
        self.browser_tabs.addTab(archive_tab, "H5 Archive")
        self._workspace_tab = workspace_tab
        self._archive_tab = archive_tab

        preview_group = QGroupBox("Frame Player")
        preview_layout = QVBoxLayout()
        preview_group.setLayout(preview_layout)
        preview_layout.addWidget(self.preview_image_label, 1)
        player_controls = QHBoxLayout()
        player_controls.addWidget(self.previous_frame_button)
        player_controls.addWidget(self.frame_slider, 1)
        player_controls.addWidget(self.next_frame_button)
        player_controls.addWidget(self.frame_position_label)
        preview_layout.addWidget(_wrap_layout(player_controls))
        preview_layout.addWidget(self.preview_info_label)

        splitter.addWidget(browser_panel)
        splitter.addWidget(preview_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.status_label)

    def _choose_directory_for_input(self, widget: QLineEdit) -> None:
        """Choose a directory and assign it to one line edit."""

        start_dir = widget.text().strip() or self._last_dir
        selected = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if not selected:
            return
        self._last_dir = selected
        widget.setText(selected)

        if widget is self.workspace_root_input:
            self.browser_tabs.setCurrentWidget(self._workspace_tab)
            self._reload_workspace_tree()
        elif widget is self.storage_root_input:
            self._refresh_metadata_summary()
            self._set_status(f"Selected storage root: {selected}")
        elif widget is self.raw_inbox_input:
            self._reload_raw_inbox()

    def _handle_storage_root_changed(self) -> None:
        """Keep dependent path suggestions aligned with the storage root."""

        self._update_growth_folder_from_inputs()
        self._refresh_metadata_summary()
        self._refresh_archive_path_preview()
        self._refresh_action_buttons()

    def _handle_workspace_root_changed(self) -> None:
        """Refresh dependent UI when the active growth folder changes."""

        self._refresh_archive_path_preview()
        self._reload_workspace_tree()
        self._refresh_action_buttons()

    def _handle_metadata_changed(self) -> None:
        """Refresh derived folder and archive naming when the metadata editor changes."""

        self._update_growth_folder_from_inputs()
        self._refresh_metadata_summary()
        self._refresh_archive_path_preview()
        self._refresh_action_buttons()

    def _set_raw_inbox_to_latest_date_folder(self) -> None:
        """Point the raw inbox to the newest date-like folder inside the storage root."""

        storage_root_text = self.storage_root_input.text().strip()
        if not storage_root_text:
            self._set_status("Choose the storage root first.")
            return

        latest_dir = _find_latest_date_folder(
            Path(storage_root_text).expanduser().resolve(),
            exclude_paths={self.workspace_root_input.text().strip()},
        )
        if latest_dir is None:
            self._set_status("No date-like recorder folder was found inside the storage root.")
            return

        self.raw_inbox_input.setText(str(latest_dir))
        self._reload_raw_inbox()
        self._set_status(f"Using recorder date folder: {latest_dir}")

    def _choose_archive_file(self) -> None:
        """Choose one packed H5 archive file."""

        start_dir = (
            self.storage_root_input.text().strip()
            or self.workspace_root_input.text().strip()
            or self._last_dir
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Packed Plume H5 Archive",
            start_dir,
            "HDF5 Files (*.h5 *.hdf5)",
        )
        if not file_path:
            return

        self._last_dir = str(Path(file_path).resolve().parent)
        self.archive_file_input.setText(file_path)
        self._load_archive_tree()

    def _load_metadata_from_dialog(self) -> None:
        """Load a PLD parameter JSON file into the editable metadata panel."""

        start_dir = self.storage_root_input.text().strip() or self._last_dir
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

        metadata_dir = str(Path(file_path).resolve().parent)
        self._metadata_file_path = file_path
        self._last_dir = metadata_dir
        self.metadata_file_input.setText(file_path)
        self.metadata_editor.setPlainText(metadata_to_text(metadata))
        self.storage_root_input.setText(metadata_dir)
        self.browser_tabs.setCurrentWidget(self._workspace_tab)
        if not self.raw_inbox_input.text().strip():
            latest_dir = _find_latest_date_folder(
                Path(metadata_dir).expanduser().resolve(),
                exclude_paths={self.workspace_root_input.text().strip()},
            )
            if latest_dir is not None:
                self.raw_inbox_input.setText(str(latest_dir))
                self._reload_raw_inbox()
        self._set_status(f"Loaded metadata JSON and set the storage root to {metadata_dir}.")

    def _save_metadata_to_dialog(self) -> None:
        """Save the edited metadata JSON to disk."""

        try:
            metadata = self._current_metadata()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Cannot save metadata: {exc}")
            return

        start_dir = (
            self.workspace_root_input.text().strip()
            or self.storage_root_input.text().strip()
            or self._last_dir
        )
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

    def _create_workspace_from_metadata(self) -> None:
        """Create the active growth folder and target folders using the loaded JSON metadata."""

        storage_root = self.storage_root_input.text().strip()
        if not storage_root:
            self._set_status("Choose a storage root directory first.")
            return

        try:
            metadata = self._current_metadata()
            growth_dir = self._suggested_growth_dir(metadata)
            result = create_plume_workspace(
                growth_dir,
                metadata,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to create plume workspace: {exc}")
            return

        self.workspace_root_input.setText(str(growth_dir))
        self._reload_workspace_tree()
        self.browser_tabs.setCurrentWidget(self._workspace_tab)
        self.workspace_tree.setFocus()
        if not self.raw_inbox_input.text().strip():
            latest_dir = _find_latest_date_folder(
                Path(storage_root).expanduser().resolve(),
                exclude_paths={str(growth_dir)},
            )
            if latest_dir is not None:
                self.raw_inbox_input.setText(str(latest_dir))
                self._reload_raw_inbox()
        self._set_status(
            f"Created or refreshed growth folder {result.root_dir} with {result.total_targets} target folders."
        )

    def _reload_raw_inbox(self) -> None:
        """Refresh the list of staged raw files waiting in the inbox folder."""

        self.raw_inbox_list.clear()
        raw_root = self.raw_inbox_input.text().strip()
        if not raw_root:
            self._refresh_action_buttons()
            return

        raw_dir = Path(raw_root).expanduser().resolve()
        if not raw_dir.exists():
            self._set_status(f"Raw inbox does not exist yet: {raw_dir}")
            self._refresh_action_buttons()
            return
        if not raw_dir.is_dir():
            self._set_status(f"Raw inbox path is not a directory: {raw_dir}")
            self._refresh_action_buttons()
            return

        file_paths = sorted(path for path in raw_dir.rglob("*") if path.is_file())
        for file_path in file_paths:
            label = str(file_path.relative_to(raw_dir)) if file_path.parent != raw_dir else file_path.name
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(file_path))
            self.raw_inbox_list.addItem(item)

        self._refresh_action_buttons()
        self._set_status(f"Raw inbox refreshed with {len(file_paths)} files.")

    def _move_selected_raw_files(self) -> None:
        """Move the selected raw inbox files into the selected workspace target's raw folder."""

        self._move_raw_files_to_target(selected_only=True)

    def _move_all_raw_files(self) -> None:
        """Move every raw inbox file into the selected workspace target folder."""

        self._move_raw_files_to_target(selected_only=False)

    def _move_raw_files_to_target(self, *, selected_only: bool) -> None:
        """Move selected or all raw inbox files into the selected target folder."""

        target_dir = self._selected_workspace_target_dir()
        if target_dir is None:
            self._set_status("Choose the destination target first, then move raw files into it.")
            return

        if selected_only:
            items = self.raw_inbox_list.selectedItems()
            empty_message = "Select one or more raw inbox files first."
        else:
            items = [self.raw_inbox_list.item(index) for index in range(self.raw_inbox_list.count())]
            empty_message = "There are no raw inbox files to move."

        if not items:
            self._set_status(empty_message)
            return

        source_paths = [item.data(Qt.UserRole) for item in items]
        if not source_paths:
            self._set_status("Select one or more raw inbox files first.")
            return

        try:
            result = stage_raw_files_for_target(source_paths, target_dir)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to move raw files: {exc}")
            return

        self._reload_raw_inbox()
        self._reload_workspace_tree()
        self.browser_tabs.setCurrentWidget(self._workspace_tab)
        self._set_status(f"Moved {result.total_files} raw files into {result.destination_dir}.")

    def _reload_workspace_tree(self) -> None:
        """Rebuild the workspace browser from the selected workspace root."""

        self.workspace_tree.clear()
        workspace_root_text = self.workspace_root_input.text().strip()
        if not workspace_root_text:
            self.workspace_summary_label.setText("Choose or create a growth folder to inspect its target structure.")
            self._refresh_target_selector([])
            self._refresh_action_buttons()
            return

        root = Path(workspace_root_text).expanduser().resolve()
        if not root.exists():
            self.workspace_summary_label.setText(f"Growth folder does not exist yet: {root}")
            self._refresh_target_selector([])
            self._clear_preview("Active growth folder does not exist yet.")
            self._refresh_action_buttons()
            return
        if not root.is_dir():
            self.workspace_summary_label.setText(f"Growth folder is not a directory: {root}")
            self._refresh_target_selector([])
            self._clear_preview("Active growth folder is not a directory.")
            self._refresh_action_buttons()
            return

        top_level_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        self._refresh_target_selector(top_level_dirs)
        selected_target_root = self.target_selector.currentData()
        for target_dir in top_level_dirs:
            target_item = QTreeWidgetItem([target_dir.name, "Target Folder", str(target_dir)])
            target_item.setData(0, ROLE_PATH, str(target_dir))
            target_item.setData(0, ROLE_KIND, "target")
            target_item.setData(0, ROLE_TARGET_ROOT, str(target_dir))
            self.workspace_tree.addTopLevelItem(target_item)
            self._add_directory_children(target_item, target_dir, target_root=target_dir)

        self.workspace_tree.expandToDepth(1)
        self.workspace_tree.resizeColumnToContents(0)
        self.workspace_tree.resizeColumnToContents(1)
        if self.workspace_tree.topLevelItemCount() > 0:
            target_item_to_select = self.workspace_tree.topLevelItem(0)
            for index in range(self.workspace_tree.topLevelItemCount()):
                item = self.workspace_tree.topLevelItem(index)
                if item.data(0, ROLE_TARGET_ROOT) == selected_target_root:
                    target_item_to_select = item
                    break
            self.workspace_tree.setCurrentItem(target_item_to_select)

        self._workspace_record = scan_plume_directory(root)
        raw_file_count = self._count_workspace_raw_files(root)
        self.workspace_summary_label.setText(
            f"Growth Folder: {root.name} | Targets: {len(top_level_dirs)} | "
            f"Raw files: {raw_file_count} | Packable targets: {self._workspace_record.total_targets} | "
            f"Plume folders: {self._workspace_record.total_plumes} | Frames: {self._workspace_record.total_frames}"
        )

        if self.workspace_tree.topLevelItemCount() == 0:
            self._clear_preview("Growth folder is empty. Create it from JSON first or choose an existing one.")
        self._refresh_action_buttons()

    def _add_directory_children(self, parent_item: QTreeWidgetItem, directory: Path, *, target_root: Path) -> None:
        """Recursively add folder-level workspace items to the tree."""

        for child_path in sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
            if child_path.name.startswith("."):
                continue

            if child_path.is_dir():
                kind = "directory"
                if child_path.name.lower() == "raw":
                    kind = "raw_dir"
                elif child_path.name == "BMP":
                    kind = "bmp_dir"
                elif child_path.parent.name == "BMP":
                    kind = "plume_dir"

                item = QTreeWidgetItem([child_path.name, self._workspace_kind_label(kind), str(child_path)])
                item.setData(0, ROLE_PATH, str(child_path))
                item.setData(0, ROLE_KIND, kind)
                item.setData(0, ROLE_TARGET_ROOT, str(target_root))
                parent_item.addChild(item)
                self._add_directory_children(item, child_path, target_root=target_root)
                continue

    def _handle_workspace_selection(self) -> None:
        """Update the preview panel based on the selected workspace tree item."""

        item = self.workspace_tree.currentItem()
        if item is None:
            return

        target_root = item.data(0, ROLE_TARGET_ROOT)
        if target_root:
            self._set_current_target_selector(str(target_root))

        item_kind = item.data(0, ROLE_KIND)
        item_path = item.data(0, ROLE_PATH)
        if not item_path:
            return

        path = Path(item_path)
        if item_kind in {"target", "bmp_dir", "plume_dir", "directory"}:
            frame_paths = self._workspace_video_frame_paths(path, item_kind)
            if frame_paths:
                self._set_workspace_video_frames(frame_paths, start_index=0)
                return

        self._clear_preview(f"{self._workspace_kind_label(item_kind)}: {path}")

    def _workspace_video_frame_paths(self, path: Path, item_kind: str) -> list[Path]:
        """Return image frames represented by one workspace tree item."""

        if not path.is_dir():
            return []
        if item_kind == "plume_dir":
            return _sorted_image_files(path.iterdir())
        if item_kind == "bmp_dir":
            direct_frames = _sorted_image_files(path.iterdir())
            if direct_frames:
                return direct_frames
        if item_kind in {"target", "bmp_dir", "directory"}:
            return _sorted_image_files(path.rglob("*"))
        return []

    def _set_workspace_video_frames(self, frame_paths: list[Path], *, start_index: int) -> None:
        """Bind the frame player to filesystem image frames."""

        self._set_video_frames(
            [{"kind": "workspace", "path": frame_path} for frame_path in frame_paths],
            start_index=start_index,
        )

    def _set_archive_video_frames(
        self,
        archive_path: Path,
        target_name: str,
        plume_index: int,
        frame_count: int,
        *,
        start_index: int,
    ) -> None:
        """Bind the frame player to one packed H5 plume sequence."""

        self._set_video_frames(
            [
                {
                    "kind": "archive",
                    "archive_path": archive_path,
                    "target_name": target_name,
                    "plume_index": plume_index,
                    "frame_index": frame_index,
                }
                for frame_index in range(frame_count)
            ],
            start_index=start_index,
        )

    def _set_video_frames(self, frame_descriptors: list[dict[str, Any]], *, start_index: int = 0) -> None:
        """Load a sequence into the frame player and show the requested frame."""

        self._video_frames = frame_descriptors
        frame_count = len(self._video_frames)
        self.frame_slider.blockSignals(True)
        if frame_count:
            self.frame_slider.setRange(0, frame_count - 1)
            self.frame_slider.setValue(max(0, min(start_index, frame_count - 1)))
        else:
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(frame_count > 1)
        self.previous_frame_button.setEnabled(frame_count > 1)
        self.next_frame_button.setEnabled(frame_count > 1)
        self.frame_slider.blockSignals(False)

        if not frame_count:
            self._clear_preview("No frames were found for the selected item.")
            return

        self._show_video_frame(self.frame_slider.value())

    def _handle_frame_slider_changed(self, frame_index: int) -> None:
        """Show the frame selected by the player slider."""

        self._show_video_frame(frame_index)

    def _step_video_frame(self, step: int) -> None:
        """Move the frame player by one relative step."""

        if not self._video_frames:
            return
        next_index = max(0, min(self.frame_slider.value() + step, len(self._video_frames) - 1))
        self.frame_slider.setValue(next_index)

    def _show_video_frame(self, frame_index: int) -> None:
        """Read and display one frame from the active player sequence."""

        if not 0 <= frame_index < len(self._video_frames):
            return

        descriptor = self._video_frames[frame_index]
        try:
            if descriptor["kind"] == "workspace":
                frame_path = Path(descriptor["path"])
                frame = read_plume_frame(frame_path)
                info_message = (
                    f"{frame_path}\n"
                    f"Frame: {frame_index + 1} of {len(self._video_frames)} | "
                    f"Shape: {frame.shape[0]} x {frame.shape[1]} px | "
                    f"Intensity range: {int(frame.min())} to {int(frame.max())}"
                )
            else:
                archive_path = Path(descriptor["archive_path"])
                target_name = str(descriptor["target_name"])
                plume_index = int(descriptor["plume_index"])
                packed_frame_index = int(descriptor["frame_index"])
                frame = read_packed_frame(archive_path, target_name, plume_index, packed_frame_index)
                info_message = (
                    f"{archive_path}\n"
                    f"Target: {target_name} | Plume: {plume_index + 1} | "
                    f"Frame: {packed_frame_index + 1} of {len(self._video_frames)}\n"
                    f"Shape: {frame.shape[0]} x {frame.shape[1]} px | "
                    f"Intensity range: {int(frame.min())} to {int(frame.max())}"
                )
        except Exception as exc:  # noqa: BLE001
            self._clear_preview(f"Failed to load frame: {exc}")
            self._set_status(f"Failed to load frame: {exc}")
            return

        self._set_preview_array(frame, info_message)
        self._refresh_frame_position_label()

    def _refresh_frame_position_label(self) -> None:
        """Show the current frame number beside the player slider."""

        if not self._video_frames:
            self.frame_position_label.setText("Frame 0 / 0")
            return
        self.frame_position_label.setText(f"Frame {self.frame_slider.value() + 1} / {len(self._video_frames)}")

    def _clear_video_player(self) -> None:
        """Reset the frame player controls."""

        self._video_frames = []
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.blockSignals(False)
        self.previous_frame_button.setEnabled(False)
        self.next_frame_button.setEnabled(False)
        self._refresh_frame_position_label()

    def _selected_workspace_target_dir(self) -> Path | None:
        """Return the target folder implied by the current workspace selection."""

        target_root = self.target_selector.currentData()
        if not target_root:
            return None
        return Path(target_root)

    def _refresh_target_selector(self, target_dirs: list[Path]) -> None:
        """Populate the destination-target selector from the current growth folder."""

        current_value = self.target_selector.currentData()
        self.target_selector.blockSignals(True)
        self.target_selector.clear()
        for target_dir in target_dirs:
            self.target_selector.addItem(target_dir.name, str(target_dir))
        self.target_selector.blockSignals(False)

        if target_dirs:
            selected_value = str(target_dirs[0])
            if current_value and any(str(path) == current_value for path in target_dirs):
                selected_value = str(current_value)
            self._set_current_target_selector(selected_value)
            return

        self.raw_target_label.setText("Destination target: none")

    def _set_current_target_selector(self, target_path: str) -> None:
        """Set the target selector to one path if that target exists in the list."""

        for index in range(self.target_selector.count()):
            if self.target_selector.itemData(index) == target_path:
                self.target_selector.blockSignals(True)
                self.target_selector.setCurrentIndex(index)
                self.target_selector.blockSignals(False)
                self.raw_target_label.setText(f"Destination target: {Path(target_path).name}")
                return

    def _handle_target_selector_changed(self) -> None:
        """Update the destination-target label when the target selector changes."""

        target_root = self.target_selector.currentData()
        if not target_root:
            self.raw_target_label.setText("Destination target: none")
            return
        self.raw_target_label.setText(f"Destination target: {Path(target_root).name}")
        for index in range(self.workspace_tree.topLevelItemCount()):
            item = self.workspace_tree.topLevelItem(index)
            if item.data(0, ROLE_TARGET_ROOT) == target_root:
                self.workspace_tree.setCurrentItem(item)
                break

    def _count_workspace_raw_files(self, root: Path) -> int:
        """Count raw-like files stored directly inside target folders."""

        total = 0
        for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            total += sum(1 for path in target_dir.iterdir() if path.is_file())
        return total

    def _load_archive_tree(self) -> None:
        """Load one packed H5 archive into the archive tree browser."""

        archive_path = self.archive_file_input.text().strip()
        if not archive_path:
            self._set_status("Choose a packed H5 archive first.")
            return

        try:
            archive_record = inspect_plume_archive(archive_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to inspect packed archive: {exc}")
            return

        self._archive_record = archive_record
        self.archive_tree.clear()

        plume_root_item = QTreeWidgetItem(
            [
                "PLD_Plumes",
                "H5 Group",
                f"{archive_record.total_targets} targets, {archive_record.total_frames} frames",
            ]
        )
        plume_root_item.setData(0, ROLE_PATH, archive_record.archive_path)
        plume_root_item.setData(0, ROLE_KIND, "archive_group")
        self.archive_tree.addTopLevelItem(plume_root_item)

        first_plume_item: QTreeWidgetItem | None = None
        for target in archive_record.targets:
            target_item = QTreeWidgetItem(
                [
                    target.target_name,
                    "Packed Target",
                    f"{target.plume_count} plumes, {target.total_frames} frames, shape {target.frame_shape}",
                ]
            )
            target_item.setData(0, ROLE_PATH, archive_record.archive_path)
            target_item.setData(0, ROLE_KIND, "archive_target")
            target_item.setData(0, ROLE_ARCHIVE_TARGET, target.target_name)
            plume_root_item.addChild(target_item)

            for plume_index, frame_count in enumerate(target.frame_counts):
                plume_item = QTreeWidgetItem(
                    [
                        f"plume_{plume_index + 1:03d}",
                        "Packed Plume",
                        f"{frame_count} frames",
                    ]
                )
                plume_item.setData(0, ROLE_PATH, archive_record.archive_path)
                plume_item.setData(0, ROLE_KIND, "archive_plume")
                plume_item.setData(0, ROLE_ARCHIVE_TARGET, target.target_name)
                plume_item.setData(0, ROLE_PLUME_INDEX, plume_index)
                plume_item.setData(0, ROLE_FRAME_COUNT, frame_count)
                target_item.addChild(plume_item)
                if first_plume_item is None:
                    first_plume_item = plume_item

        self.archive_tree.expandToDepth(1)
        self.archive_tree.resizeColumnToContents(0)
        self.archive_tree.resizeColumnToContents(1)

        self.archive_summary_label.setText(self._format_archive_summary_label(archive_record))
        self.browser_tabs.setCurrentWidget(self._archive_tab)
        self._set_status(f"Loaded packed archive {archive_record.archive_path}.")

        if first_plume_item is not None:
            self.archive_tree.setCurrentItem(first_plume_item)
        else:
            self._clear_preview("Archive contains no packed frames.")

    def _handle_archive_selection(self) -> None:
        """Update the preview panel based on the selected H5 tree item."""

        item = self.archive_tree.currentItem()
        if item is None:
            return

        item_kind = item.data(0, ROLE_KIND)
        archive_path = item.data(0, ROLE_PATH)
        if item_kind == "archive_target" and archive_path and item.childCount() > 0:
            item = item.child(0)
            item_kind = item.data(0, ROLE_KIND)

        if item_kind == "archive_plume" and archive_path:
            self._set_archive_video_frames(
                Path(archive_path),
                item.data(0, ROLE_ARCHIVE_TARGET),
                int(item.data(0, ROLE_PLUME_INDEX)),
                int(item.data(0, ROLE_FRAME_COUNT)),
                start_index=0,
            )
            return

        if item_kind == "archive_frame" and archive_path:
            plume_item = item.parent()
            frame_count = int(plume_item.data(0, ROLE_FRAME_COUNT)) if plume_item is not None else 1
            self._set_archive_video_frames(
                Path(archive_path),
                item.data(0, ROLE_ARCHIVE_TARGET),
                int(item.data(0, ROLE_PLUME_INDEX)),
                frame_count,
                start_index=int(item.data(0, ROLE_FRAME_INDEX)),
            )
            return

        self._clear_preview(f"{item.text(1)}: {item.text(2)}")

    def _preview_image_file(self, file_path: Path) -> None:
        """Preview one filesystem image stored inside the workspace tree."""

        try:
            frame = read_plume_frame(file_path)
        except Exception as exc:  # noqa: BLE001
            self._clear_preview(f"Failed to load image preview: {exc}")
            self._set_status(f"Failed to load image preview: {exc}")
            return

        self._set_preview_array(
            frame,
            f"{file_path}\n"
            f"Shape: {frame.shape[0]} x {frame.shape[1]} px | "
            f"Intensity range: {int(frame.min())} to {int(frame.max())}",
        )

    def _preview_archive_frame(
        self,
        archive_path: Path,
        target_name: str,
        plume_index: int,
        frame_index: int,
    ) -> None:
        """Preview one frame loaded from a packed H5 archive."""

        try:
            frame = read_packed_frame(archive_path, target_name, plume_index, frame_index)
        except Exception as exc:  # noqa: BLE001
            self._clear_preview(f"Failed to load packed frame: {exc}")
            self._set_status(f"Failed to load packed frame: {exc}")
            return

        self._set_preview_array(
            frame,
            f"{archive_path}\n"
            f"Target: {target_name} | Plume: {plume_index + 1} | Frame: {frame_index + 1}\n"
            f"Shape: {frame.shape[0]} x {frame.shape[1]} px | "
            f"Intensity range: {int(frame.min())} to {int(frame.max())}",
        )

    def _set_preview_array(self, frame, info_message: str) -> None:
        """Convert one grayscale array into the preview pixmap."""

        image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
        self._preview_pixmap = QPixmap.fromImage(image)
        self._refresh_preview_pixmap()
        self.preview_info_label.setText(info_message)

    def _refresh_preview_pixmap(self) -> None:
        """Scale the loaded preview image to the current preview panel size."""

        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return

        scaled = self._preview_pixmap.scaled(
            self.preview_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_image_label.setPixmap(scaled)
        self.preview_image_label.setText("")

    def resizeEvent(self, event):  # noqa: N802
        """Keep the preview image scaled when the window size changes."""

        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def _clear_preview(self, info_message: str) -> None:
        """Reset the preview panel to a descriptive text-only state."""

        self._clear_video_player()
        self._preview_pixmap = None
        self.preview_image_label.setPixmap(QPixmap())
        self.preview_image_label.setText("Select a plume folder, BMP folder, or H5 plume to preview it here.")
        self.preview_info_label.setText(info_message)

    def _pack_archive(self) -> None:
        """Create the HDF5 plume archive and update the on-screen summary."""

        try:
            output_path = self._archive_output_path()
            metadata = self._optional_metadata()
            result = pack_plume_directory(self.workspace_root_input.text(), output_path, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Pack failed: {exc}")
            return

        self.archive_file_input.setText(result.output_path)
        self._refresh_archive_path_preview()
        self._set_status(
            f"Packed {result.total_targets} targets and {result.total_frames} frames into {result.output_path}."
        )
        self._load_archive_tree()
        self._reload_workspace_tree()

    def _archive_output_path(self) -> str:
        """Return the full HDF5 output path derived from the current form fields."""

        workspace_root = self.workspace_root_input.text().strip()
        if not workspace_root:
            raise ValueError("Choose or create a growth folder first.")

        workspace_path = Path(workspace_root).expanduser().resolve()
        output_dir = Path(self.storage_root_input.text().strip()).expanduser().resolve() if self.storage_root_input.text().strip() else workspace_path.parent
        return str(output_dir / f"{workspace_path.name}.h5")

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

    def _refresh_archive_path_preview(self) -> None:
        """Show the full output path that will be used for the packed H5 file."""

        workspace_root = self.workspace_root_input.text().strip()
        if not workspace_root:
            self.archive_path_preview_label.setText("Choose or create a growth folder to preview the packed H5 path.")
            return

        workspace_path = Path(workspace_root).expanduser().resolve()
        if self.storage_root_input.text().strip():
            output_dir = Path(self.storage_root_input.text().strip()).expanduser().resolve()
        else:
            output_dir = workspace_path.parent
        output_path = output_dir / f"{workspace_path.name}.h5"
        self.archive_path_preview_label.setText(str(output_path))

    def _update_growth_folder_from_inputs(self) -> None:
        """Refresh the suggested active growth folder from storage-root and metadata inputs."""

        storage_root = self.storage_root_input.text().strip()
        if not storage_root:
            return

        try:
            metadata = self._optional_metadata()
        except ValueError:
            return

        suggested_growth_dir = self._suggested_growth_dir_path_text(metadata)
        if not suggested_growth_dir:
            return
        current_growth_dir = self.workspace_root_input.text().strip()
        if not current_growth_dir or current_growth_dir == self._last_suggested_growth_dir:
            self.workspace_root_input.setText(suggested_growth_dir)
        self._last_suggested_growth_dir = suggested_growth_dir

    def _suggested_growth_dir(self, metadata: dict[str, Any] | None) -> Path:
        """Return the growth folder that should be created inside the storage root."""

        growth_dir_text = self._suggested_growth_dir_path_text(metadata)
        if not growth_dir_text:
            raise ValueError("Load metadata with header fields before creating the growth folder.")
        return Path(growth_dir_text)

    def _suggested_growth_dir_path_text(self, metadata: dict[str, Any] | None = None) -> str:
        """Build the full suggested active-growth-folder path as text."""

        storage_root = self.storage_root_input.text().strip()
        if not storage_root:
            return ""
        if not _metadata_supports_growth_naming(metadata):
            return ""
        growth_stem = build_plume_growth_stem(storage_root, metadata=metadata)
        return str(Path(storage_root).expanduser().resolve() / growth_stem)

    def _refresh_metadata_summary(self) -> None:
        """Show a compact preview of the growth folder and target folders from the loaded JSON."""

        try:
            metadata = self._optional_metadata()
        except ValueError as exc:
            self.metadata_summary_label.setText(f"Metadata JSON is invalid: {exc}")
            return

        if not _metadata_supports_growth_naming(metadata):
            self.metadata_summary_label.setText(
                "Load a JSON file with header fields such as Growth ID, User Name, and Date to preview the growth folder."
            )
            return

        growth_dir_text = self._suggested_growth_dir_path_text(metadata)
        try:
            target_defs = build_plume_workspace_targets(
                metadata or {},
            )
        except Exception as exc:  # noqa: BLE001
            self.metadata_summary_label.setText(f"Metadata loaded, but target folders could not be previewed: {exc}")
            return

        preview_names = [target.folder_name for target in target_defs[:4]]
        extra_count = max(0, len(target_defs) - len(preview_names))
        target_preview = ", ".join(preview_names) if preview_names else "none"
        if extra_count:
            target_preview = f"{target_preview}, +{extra_count} more"

        self.metadata_summary_label.setText(
            f"Growth folder to create: {Path(growth_dir_text).name if growth_dir_text else 'not ready'} | "
            f"Target folders: {target_preview}"
        )

    def _refresh_action_buttons(self) -> None:
        """Enable or disable action buttons based on the current workflow state."""

        try:
            metadata = self._optional_metadata()
        except ValueError:
            metadata = None

        storage_root_ready = bool(self.storage_root_input.text().strip())
        metadata_ready = _metadata_supports_growth_naming(metadata)
        workspace_root = self.workspace_root_input.text().strip()
        workspace_ready = Path(workspace_root).expanduser().is_dir() if workspace_root else False
        raw_inbox_ready = self.raw_inbox_list.count() > 0
        target_ready = self.target_selector.count() > 0

        self.create_workspace_button.setEnabled(storage_root_ready and metadata_ready)
        self.pack_button.setEnabled(workspace_ready)
        self.move_raw_button.setEnabled(workspace_ready and target_ready and raw_inbox_ready)
        self.move_all_raw_button.setEnabled(workspace_ready and target_ready and raw_inbox_ready)

    def _format_workspace_creation_result(self, result: PlumeWorkspaceCreationResult) -> str:
        """Format a readable summary for workspace creation."""

        lines = [
            f"workspace_root: {result.root_dir}",
            f"targets_created: {result.total_targets}",
            "",
            "Target Folders:",
        ]
        for folder in result.target_folders:
            suffix = " (pre-ablation)" if folder.is_pre_ablation else ""
            lines.append(f"- {folder.folder_name}{suffix}")
            lines.append(f"  target_dir: {folder.target_dir}")
        return "\n".join(lines)

    def _format_raw_staging_result(self, result: RawFileStagingResult) -> str:
        """Format a readable summary for raw-file staging."""

        lines = [
            f"target_dir: {result.target_dir}",
            f"destination_dir: {result.destination_dir}",
            f"files_moved: {result.total_files}",
            "",
            "Moved Files:",
        ]
        lines.extend(f"- {path}" for path in result.moved_files)
        return "\n".join(lines)

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

    def _format_archive_summary_label(self, archive_record: PlumeArchiveRecord) -> str:
        """Build the compact summary label shown above the H5 tree."""

        metadata_note = "yes" if archive_record.metadata_json else "no"
        return (
            f"Archive: {archive_record.archive_path}\n"
            f"Created: {archive_record.created_at or 'unknown'} | "
            f"Source dir: {archive_record.source_dir or 'unknown'}\n"
            f"Targets: {archive_record.total_targets} | "
            f"Plume folders: {archive_record.total_plumes} | "
            f"Frames: {archive_record.total_frames} | "
            f"Metadata stored: {metadata_note}"
        )

    def _format_archive_record(self, archive_record: PlumeArchiveRecord) -> str:
        """Format a readable archive summary for the result panel."""

        lines = [
            f"archive_path: {archive_record.archive_path}",
            f"source_dir: {archive_record.source_dir}",
            f"created_at: {archive_record.created_at}",
            f"targets: {archive_record.total_targets}",
            f"plume_folders: {archive_record.total_plumes}",
            f"frames: {archive_record.total_frames}",
            f"metadata_json_stored: {'yes' if archive_record.metadata_json else 'no'}",
            "",
            "Target Details:",
        ]
        for target in archive_record.targets:
            lines.append(
                f"- {target.target_name}: {target.plume_count} plume folders, "
                f"{target.total_frames} frames, frame shape {target.frame_shape}"
            )
        return "\n".join(lines)

    def _workspace_kind_label(self, kind: str) -> str:
        """Return a short user-facing label for one workspace tree item kind."""

        label_map = {
            "bmp_dir": "BMP Folder",
            "directory": "Folder",
            "file": "File",
            "image_file": "Image File",
            "plume_dir": "Plume Folder",
            "raw_dir": "Raw Folder",
            "target": "Target Folder",
        }
        return label_map.get(kind, kind)

    def _set_status(self, message: str) -> None:
        """Update the status text shown at the bottom of the window."""

        self.status_label.setText(message)


def _find_latest_date_folder(root: Path, *, exclude_paths: set[str] | None = None) -> Path | None:
    """Return the newest date-like child folder below one storage root."""

    if not root.is_dir():
        return None

    excluded = {
        str(Path(path).expanduser().resolve())
        for path in (exclude_paths or set())
        if path
    }
    dated_children: list[tuple[datetime, float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if str(child.resolve()) in excluded:
            continue

        parsed_date = _parse_folder_date(child.name)
        if parsed_date is None:
            continue
        dated_children.append((parsed_date, child.stat().st_mtime, child))

    if not dated_children:
        return None

    dated_children.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return dated_children[0][2]


def _parse_folder_date(folder_name: str) -> datetime | None:
    """Parse common recorder date-folder naming patterns."""

    normalized = folder_name.strip()
    for pattern in ("%m%d%Y", "%m-%d-%Y", "%m_%d_%Y", "%Y%m%d", "%Y-%m-%d", "%Y_%m_%d"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    return None


def _metadata_supports_growth_naming(metadata: dict[str, Any] | None) -> bool:
    """Return True when metadata contains enough header information to name a growth folder."""

    if not isinstance(metadata, dict):
        return False
    header = metadata.get("header", {})
    if not isinstance(header, dict):
        return False
    return any(str(header.get(key, "")).strip() for key in ("Sample Name", "Growth ID", "User Name", "Date"))


def _sorted_image_files(paths) -> list[Path]:
    """Return image files sorted by path name for stable frame playback."""

    return sorted(
        (path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


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
    app.setFont(custom_font, "QListWidget")
    app.setFont(custom_font, "QPlainTextEdit")
    app.setFont(custom_font, "QTreeWidget")

    window = PlumeManagerWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

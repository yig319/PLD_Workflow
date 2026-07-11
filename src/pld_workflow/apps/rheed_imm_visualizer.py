"""Standalone RHEED IMM visualizer for lightweight frame inspection.

This app is intentionally separate from the common PLD parameter recorder.
It focuses on one task: inspect a large `.imm` movie file and load only the
frame you ask for, rather than reading the entire movie into memory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..rheed_imm import ImmInfo, ImmMovie


def crop_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Return a cropped view using `(y_min, y_max, x_min, x_max)` coordinates."""

    y_min, y_max, x_min, x_max = roi
    return np.asarray(frame)[y_min:y_max, x_min:x_max]


def _format_float(value: float | None, *, digits: int = 3, suffix: str = "") -> str:
    """Format optional floating-point values for status and metadata labels."""

    if value is None:
        return "Not set"
    return f"{value:.{digits}f}{suffix}"


def _normalize_frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a numeric frame into a viewable 8-bit grayscale image.

    Percentile scaling makes single-frame previews easier to read because
    bright diffraction spots do not completely flatten the rest of the image.
    """

    array = np.asarray(frame, dtype=float)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.5))
    if high <= low:
        high = float(finite.max())
        low = float(finite.min())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    scaled = (array - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Create a Qt grayscale image from one detector frame."""

    image_data = _normalize_frame_to_uint8(frame)
    height, width = image_data.shape
    image = QImage(image_data.data, width, height, image_data.strides[0], QImage.Format_Grayscale8)
    return image.copy()


class ImmDropBlock(QGroupBox):
    """Drag/drop block for selecting one IMM movie file."""

    file_selected = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, start_dir_provider, parent=None):
        super().__init__("RHEED IMM Movie", parent)
        self._start_dir_provider = start_dir_provider
        self.setAcceptDrops(True)
        self.setMinimumSize(260, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.label = QLabel("Drag a .imm file here or click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.label.setStyleSheet(
            "border: 2px dashed #8aa7c2; border-radius: 8px; background: #f6fbff; padding: 12px;"
        )
        self.label.setMinimumHeight(120)

        self.path_label = QLabel("No file selected")
        self.path_label.setWordWrap(True)
        self.path_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.path_label.setStyleSheet("color: #4d607a;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.path_label)
        self.setLayout(layout)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._open_dialog()
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            local_file = url.toLocalFile()
            if local_file.lower().endswith(".imm"):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            local_file = url.toLocalFile()
            if local_file.lower().endswith(".imm"):
                self.file_selected.emit(local_file)
                event.acceptProposedAction()
                return
        self.status_message.emit("Please drop a valid .imm movie file.")
        event.ignore()

    def set_file_path(self, file_path: str) -> None:
        """Update the visible file-path label after selection."""

        self.path_label.setText(file_path)

    def _open_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load RHEED IMM Movie",
            self._start_dir_provider(),
            "IMM Files (*.imm);;All Files (*)",
        )
        if file_path:
            self.file_selected.emit(file_path)


class CropImageView(QWidget):
    """Image panel that supports interactive rectangular crop selection."""

    roi_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def clear(self) -> None:
        """Reset the current frame and crop selection."""

        self._frame = None
        self._pixmap = None
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_frame(self, frame: np.ndarray) -> None:
        """Display a new frame and clear any old crop selection."""

        self._frame = np.asarray(frame)
        self._pixmap = QPixmap.fromImage(frame_to_qimage(self._frame))
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def clear_roi(self) -> None:
        """Remove the current crop selection while keeping the loaded frame."""

        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def current_roi(self) -> tuple[int, int, int, int] | None:
        """Return the current crop coordinates in image pixels."""

        return self._roi

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "Load one frame to inspect and crop it.")
            return

        target_rect = self._image_target_rect()
        painter.drawPixmap(target_rect, self._pixmap)

        selection = self._current_display_rect()
        if selection is not None:
            painter.setPen(QPen(Qt.cyan, 2))
            painter.drawRect(selection)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton or self._frame is None:
            return super().mousePressEvent(event)
        if not self._image_target_rect().contains(event.pos()):
            return
        self._drag_start = event.pos()
        self._drag_current = event.pos()
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_start is None:
            return super().mouseMoveEvent(event)
        self._drag_current = self._clamp_to_image_rect(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton or self._drag_start is None or self._frame is None:
            return super().mouseReleaseEvent(event)

        self._drag_current = self._clamp_to_image_rect(event.pos())
        roi = self._drag_points_to_roi()
        self._drag_start = None
        self._drag_current = None
        if roi is None:
            self._roi = None
            self.update()
            return

        self._roi = roi
        self.update()
        self.roi_changed.emit(roi)

    def _image_target_rect(self) -> QRect:
        """Return the rectangle used to draw the current image inside the widget."""

        if self._pixmap is None or self._pixmap.isNull():
            return QRect()

        contents = self.rect().adjusted(8, 8, -8, -8)
        scaled_size = self._pixmap.size().scaled(contents.size(), Qt.KeepAspectRatio)
        x_offset = contents.x() + max(0, (contents.width() - scaled_size.width()) // 2)
        y_offset = contents.y() + max(0, (contents.height() - scaled_size.height()) // 2)
        return QRect(x_offset, y_offset, scaled_size.width(), scaled_size.height())

    def _clamp_to_image_rect(self, point: QPoint) -> QPoint:
        """Clamp one mouse position into the visible image rectangle."""

        rect = self._image_target_rect()
        if rect.isNull():
            return point
        x_value = min(max(point.x(), rect.left()), rect.right())
        y_value = min(max(point.y(), rect.top()), rect.bottom())
        return QPoint(x_value, y_value)

    def _drag_points_to_roi(self) -> tuple[int, int, int, int] | None:
        """Convert the current drag rectangle into image-pixel crop coordinates."""

        if self._frame is None or self._drag_start is None or self._drag_current is None:
            return None

        start = self._widget_point_to_image(self._drag_start)
        end = self._widget_point_to_image(self._drag_current)
        if start is None or end is None:
            return None

        x0 = min(start[0], end[0])
        x1 = max(start[0], end[0]) + 1
        y0 = min(start[1], end[1])
        y1 = max(start[1], end[1]) + 1
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            return None
        return (y0, y1, x0, x1)

    def _widget_point_to_image(self, point: QPoint) -> tuple[int, int] | None:
        """Map one widget-space point into image pixel coordinates."""

        if self._frame is None:
            return None
        rect = self._image_target_rect()
        if rect.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return None

        width = self._frame.shape[1]
        height = self._frame.shape[0]
        x_ratio = (point.x() - rect.x()) / float(rect.width())
        y_ratio = (point.y() - rect.y()) / float(rect.height())
        x_value = int(np.clip(x_ratio * width, 0, width - 1))
        y_value = int(np.clip(y_ratio * height, 0, height - 1))
        return x_value, y_value

    def _roi_to_display_rect(self, roi: tuple[int, int, int, int]) -> QRect:
        """Convert image-pixel crop coordinates back to widget display coordinates."""

        if self._frame is None:
            return QRect()

        y0, y1, x0, x1 = roi
        image_rect = self._image_target_rect()
        width = self._frame.shape[1]
        height = self._frame.shape[0]

        left = image_rect.x() + int(round((x0 / float(width)) * image_rect.width()))
        right = image_rect.x() + int(round((x1 / float(width)) * image_rect.width()))
        top = image_rect.y() + int(round((y0 / float(height)) * image_rect.height()))
        bottom = image_rect.y() + int(round((y1 / float(height)) * image_rect.height()))
        return QRect(QPoint(left, top), QPoint(right, bottom)).normalized()

    def _current_display_rect(self) -> QRect | None:
        """Return the selection rectangle currently shown to the user."""

        if self._drag_start is not None and self._drag_current is not None:
            return QRect(self._drag_start, self._drag_current).normalized()
        if self._roi is not None:
            return self._roi_to_display_rect(self._roi)
        return None


class RheedImmVisualizerWindow(QWidget):
    """Window for inspecting one IMM movie and loading one frame at a time."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD RHEED IMM Visualizer")
        self.setMinimumSize(1420, 860)

        self._last_dir = os.getcwd()
        self._movie: ImmMovie | None = None
        self._info: ImmInfo | None = None
        self._current_frame: np.ndarray | None = None
        self._current_frame_index: int | None = None

        self.drop_block = ImmDropBlock(self._start_dir)
        self.drop_block.file_selected.connect(self._load_movie_file)
        self.drop_block.status_message.connect(self._set_status)

        self.status_label = QLabel("Load a .imm movie to inspect metadata first, then load one frame on demand.")
        self.status_label.setWordWrap(True)

        self.file_label = QLabel("Not loaded")
        self.file_label.setWordWrap(True)
        self.format_label = QLabel("IMM")
        self.signature_label = QLabel("Not loaded")
        self.frame_count_label = QLabel("Not loaded")
        self.frame_size_label = QLabel("Not loaded")
        self.dtype_label = QLabel("Not loaded")
        self.frame_stride_label = QLabel("Not loaded")
        self.fps_label = QLabel("Not set")
        self.duration_label = QLabel("Not set")
        self.trailing_label = QLabel("Not loaded")

        metadata_box = QGroupBox("Movie Information")
        metadata_layout = QFormLayout()
        metadata_layout.addRow("File", self.file_label)
        metadata_layout.addRow("Format", self.format_label)
        metadata_layout.addRow("Signature", self.signature_label)
        metadata_layout.addRow("Frame count", self.frame_count_label)
        metadata_layout.addRow("Frame size", self.frame_size_label)
        metadata_layout.addRow("Pixel dtype", self.dtype_label)
        metadata_layout.addRow("Frame stride / header", self.frame_stride_label)
        metadata_layout.addRow("Movie fps", self.fps_label)
        metadata_layout.addRow("Duration", self.duration_label)
        metadata_layout.addRow("Trailing bytes", self.trailing_label)
        metadata_box.setLayout(metadata_layout)

        timing_box = QGroupBox("Timing Inputs")
        timing_layout = QFormLayout()
        self.fps_input = QDoubleSpinBox()
        self.fps_input.setRange(0.001, 100000.0)
        self.fps_input.setDecimals(4)
        self.fps_input.setValue(50.0)
        self.fps_input.setToolTip("Movie frame rate in frames/second. Used to convert time into frame index.")
        self.fps_input.valueChanged.connect(self._refresh_movie_timing)

        self.laser_rate_input = QDoubleSpinBox()
        self.laser_rate_input.setRange(0.001, 100000.0)
        self.laser_rate_input.setDecimals(4)
        self.laser_rate_input.setValue(10.0)
        self.laser_rate_input.setToolTip("Laser repetition rate in pulses/second. Used to convert pulse count into time.")

        timing_layout.addRow("Movie fps", self.fps_input)
        timing_layout.addRow("Laser rep rate (Hz)", self.laser_rate_input)
        timing_box.setLayout(timing_layout)

        load_box = QGroupBox("Load One Frame")
        load_layout = QGridLayout()
        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.0, 1_000_000.0)
        self.time_input.setDecimals(6)
        self.time_input.setSingleStep(0.1)
        self.time_button = QPushButton("Load By Time (s)")
        self.time_button.clicked.connect(self._load_frame_by_time)

        self.pulse_input = QDoubleSpinBox()
        self.pulse_input.setRange(0.0, 1_000_000_000.0)
        self.pulse_input.setDecimals(3)
        self.pulse_input.setSingleStep(1.0)
        self.pulse_button = QPushButton("Load By Pulse Count")
        self.pulse_button.clicked.connect(self._load_frame_by_pulses)

        self.frame_summary_label = QLabel("No frame loaded")
        self.frame_summary_label.setWordWrap(True)

        load_layout.addWidget(QLabel("Elapsed time (s)"), 0, 0)
        load_layout.addWidget(self.time_input, 0, 1)
        load_layout.addWidget(self.time_button, 0, 2)
        load_layout.addWidget(QLabel("Pulse count"), 1, 0)
        load_layout.addWidget(self.pulse_input, 1, 1)
        load_layout.addWidget(self.pulse_button, 1, 2)
        load_layout.addWidget(self.frame_summary_label, 2, 0, 1, 3)
        load_box.setLayout(load_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.drop_block)
        left_layout.addWidget(metadata_box)
        left_layout.addWidget(timing_box)
        left_layout.addWidget(load_box)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(430)

        self.frame_view = CropImageView()
        self.frame_view.roi_changed.connect(self._update_crop_preview)

        frame_box = QGroupBox("Loaded Frame")
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.frame_view)
        frame_box.setLayout(frame_layout)

        self.crop_info_label = QLabel("Draw a rectangle on the loaded frame to create a crop.")
        self.crop_info_label.setWordWrap(True)

        self.crop_preview_label = QLabel("Crop preview will appear here.")
        self.crop_preview_label.setAlignment(Qt.AlignCenter)
        self.crop_preview_label.setMinimumSize(QSize(320, 240))
        self.crop_preview_label.setStyleSheet("background: #f8fafc; border: 1px solid #d7e0ea;")

        self.clear_crop_button = QPushButton("Clear Crop")
        self.clear_crop_button.clicked.connect(self._clear_crop)
        self.copy_image_button = QPushButton("Copy Image")
        self.copy_image_button.clicked.connect(self._copy_current_image_to_clipboard)
        self.export_image_button = QPushButton("Export Image")
        self.export_image_button.clicked.connect(self._export_current_image)

        crop_box = QGroupBox("Crop Preview")
        crop_layout = QVBoxLayout()
        crop_layout.addWidget(self.crop_preview_label, 1)
        crop_layout.addWidget(self.crop_info_label)

        crop_button_row = QHBoxLayout()
        crop_button_row.addWidget(self.clear_crop_button)
        crop_button_row.addWidget(self.copy_image_button)
        crop_button_row.addWidget(self.export_image_button)
        crop_layout.addLayout(crop_button_row)
        crop_box.setLayout(crop_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(frame_box, 3)
        right_layout.addWidget(crop_box, 2)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)

        root_layout = QHBoxLayout()
        root_layout.addWidget(left_panel)
        root_layout.addWidget(right_panel, 1)
        self.setLayout(root_layout)

    def _start_dir(self) -> str:
        return self._last_dir

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _load_movie_file(self, file_path: str) -> None:
        """Inspect one IMM file and refresh the metadata panel."""

        try:
            movie = ImmMovie(file_path, fps=self.fps_input.value())
            info = movie.inspect()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to inspect IMM file: {exc}")
            return

        self._movie = movie
        self._info = info
        self._last_dir = str(Path(file_path).resolve().parent)
        self.drop_block.set_file_path(file_path)
        self._update_metadata_labels()

        self._current_frame = None
        self._current_frame_index = None
        self.frame_view.clear()
        self._clear_crop_preview_text()
        self.frame_summary_label.setText("No frame loaded")
        self._set_status("IMM file inspected. Adjust movie fps if needed, then load a frame by time or pulse count.")

    def _refresh_movie_timing(self) -> None:
        """Rebuild the movie object when the user changes the assumed fps."""

        if self._info is None:
            self.fps_label.setText(_format_float(self.fps_input.value(), digits=4, suffix=" frames/s"))
            self.duration_label.setText("Not loaded")
            return

        try:
            self._movie = ImmMovie(self._info.path, fps=self.fps_input.value())
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to refresh movie timing: {exc}")
            return

        self._update_metadata_labels()

    def _update_metadata_labels(self) -> None:
        """Fill the metadata panel using the currently loaded IMM file."""

        if self._info is None or self._movie is None:
            return

        self.file_label.setText(str(self._info.path))
        self.signature_label.setText(self._info.signature)
        self.frame_count_label.setText(str(self._info.frame_count))
        self.frame_size_label.setText(f"{self._info.height} x {self._info.width} pixels")
        self.dtype_label.setText(self._info.dtype)
        self.frame_stride_label.setText(
            f"{self._info.frame_stride_bytes} B / {self._info.header_bytes} B"
        )
        self.fps_label.setText(_format_float(self._movie.fps, digits=4, suffix=" frames/s"))
        self.duration_label.setText(_format_float(self._movie.duration_s, digits=3, suffix=" s"))
        self.trailing_label.setText(f"{self._info.trailing_bytes} B")

    def _require_movie(self) -> ImmMovie | None:
        """Return the current movie, or show a helpful status message first."""

        if self._movie is None:
            self._set_status("Load an .imm movie first.")
            return None
        return self._movie

    def _load_frame_by_time(self) -> None:
        """Load exactly one frame using the requested elapsed time."""

        movie = self._require_movie()
        if movie is None:
            return

        time_s = self.time_input.value()
        try:
            frame_index, frame = movie.load_frame_by_time(time_s, as_float=True)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to load frame by time: {exc}")
            return

        self._set_loaded_frame(frame_index, frame, source_label=f"time {time_s:.6f} s")

    def _load_frame_by_pulses(self) -> None:
        """Load exactly one frame using the requested pulse count."""

        movie = self._require_movie()
        if movie is None:
            return

        pulse_count = self.pulse_input.value()
        laser_rate_hz = self.laser_rate_input.value()
        try:
            frame_index, frame = movie.load_frame_by_pulse_count(
                pulse_count,
                laser_rate_hz=laser_rate_hz,
                as_float=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Failed to load frame by pulse count: {exc}")
            return

        self._set_loaded_frame(frame_index, frame, source_label=f"pulse count {pulse_count:.3f}")

    def _set_loaded_frame(self, frame_index: int, frame: np.ndarray, *, source_label: str) -> None:
        """Update the frame viewer, summary labels, and crop preview after one load."""

        if self._movie is None:
            return

        self._current_frame = np.asarray(frame)
        self._current_frame_index = int(frame_index)
        self.frame_view.set_frame(self._current_frame)
        self._clear_crop_preview_text()

        time_s = self._movie.time_from_frame_index(self._current_frame_index)
        pulse_count = self._movie.pulse_count_from_frame_index(
            self._current_frame_index,
            self.laser_rate_input.value(),
        )
        self.frame_summary_label.setText(
            "Loaded frame "
            f"{self._current_frame_index} from {source_label}. "
            f"Mapped time: {time_s:.6f} s. "
            f"Estimated pulse count: {pulse_count:.3f}."
        )
        self._set_status(
            f"Loaded frame {self._current_frame_index} without reading the full movie into memory."
        )

    def _update_crop_preview(self, roi: tuple[int, int, int, int]) -> None:
        """Refresh the crop preview panel after the user draws a new ROI."""

        if self._current_frame is None:
            return

        crop = crop_frame(self._current_frame, roi)
        pixmap = QPixmap.fromImage(frame_to_qimage(crop))
        scaled = pixmap.scaled(
            self.crop_preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.crop_preview_label.setPixmap(scaled)
        y0, y1, x0, x1 = roi
        self.crop_info_label.setText(
            f"Crop ROI: y={y0}:{y1}, x={x0}:{x1} | size: {y1 - y0} x {x1 - x0} pixels"
        )

    def _clear_crop(self) -> None:
        """Clear the ROI selection and the crop preview panel."""

        self.frame_view.clear_roi()
        self._clear_crop_preview_text()
        self._set_status("Crop selection cleared.")

    def _clear_crop_preview_text(self) -> None:
        """Reset the crop-preview widgets to their default instructional text."""

        self.crop_preview_label.setPixmap(QPixmap())
        self.crop_preview_label.setText("Crop preview will appear here.")
        self.crop_info_label.setText("Draw a rectangle on the loaded frame to create a crop.")

    def _current_image_array(self) -> tuple[np.ndarray | None, str]:
        """Return the active image data plus a short label for status messages.

        When a crop exists, image actions operate on the crop first because
        that is usually the focused region the user is working with. If no ROI
        is selected, the full loaded frame is used instead.
        """

        if self._current_frame is None:
            return None, "image"

        roi = self.frame_view.current_roi()
        if roi is None:
            return self._current_frame, "full frame"
        return crop_frame(self._current_frame, roi), "cropped image"

    def _copy_current_image_to_clipboard(self) -> None:
        """Copy the active image to the clipboard.

        The active image is the crop when a crop exists, otherwise the full
        loaded frame. This restores the simpler "Copy Image" behavior from the
        earlier interface while still supporting crop-focused work.
        """

        image_array, image_label = self._current_image_array()
        if image_array is None:
            self._set_status("Load a frame before copying an image.")
            return

        image = frame_to_qimage(image_array)
        QGuiApplication.clipboard().setImage(image)
        self._set_status(f"{image_label.capitalize()} copied to clipboard.")

    def _export_current_image(self) -> None:
        """Export the active image to disk as a standard image file."""

        image_array, image_label = self._current_image_array()
        if image_array is None:
            self._set_status("Load a frame before exporting an image.")
            return

        default_dir = self._last_dir
        if self._movie is not None:
            default_dir = str(self._movie.path.parent)

        default_name = "rheed_frame.png"
        if self._current_frame_index is not None:
            default_name = f"rheed_frame_{self._current_frame_index:06d}.png"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(Path(default_dir) / default_name),
            "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;BMP Image (*.bmp);;All Files (*)",
        )
        if not save_path:
            return

        image = frame_to_qimage(image_array)
        if not image.save(save_path):
            self._set_status(f"Failed to export {image_label}.")
            return

        self._set_status(f"{image_label.capitalize()} exported to {save_path}.")


def main() -> int:
    """Launch the standalone RHEED IMM visualizer."""

    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)
    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QLabel")

    window = RheedImmVisualizerWindow()
    window.show()
    return app.exec_()


__all__ = ["RheedImmVisualizerWindow", "main"]

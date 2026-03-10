"""Raw XRD/AFM visualization helpers using XRD-utils and AFM-tools."""

from __future__ import annotations

import os
import sys
import types
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


@dataclass
class RawVisualizationResult:
    """Result of one raw-data visualization call."""

    backend: str
    preview_pixmap: Optional[QPixmap]
    message: str


class ClickablePreviewLabel(QLabel):
    """Preview label that emits a signal when clicked."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):  # noqa: N802 (Qt API name)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class RawDataDropBlock(QGroupBox):
    """Square drag/drop + click block for raw data files."""

    file_selected = pyqtSignal(str, str)
    status_message = pyqtSignal(str)

    def __init__(self, raw_type: str, title: str, start_dir_provider: Callable[[], str], parent=None):
        super().__init__(title, parent)
        self.raw_type = raw_type
        self._start_dir_provider = start_dir_provider
        self._original_pixmap: Optional[QPixmap] = None
        self._loaded_file_path: Optional[str] = None

        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 280)

        self.preview_label = ClickablePreviewLabel(self._drag_hint())
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.clicked.connect(self._open_file_dialog)
        self.preview_label.setStyleSheet(
            "border: 2px dashed #9ab0c7; border-radius: 8px; background: #f7fbff; padding: 8px;"
        )
        self.preview_label.setMinimumHeight(180)
        self.preview_label.setCursor(Qt.PointingHandCursor)

        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #4d607a;")

        self.copy_button = QPushButton("Copy Image")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_preview_to_clipboard)

        self.export_button = QPushButton("Export PNG")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_preview_image)

        button_row = QHBoxLayout()
        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.export_button)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.path_label)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt API name)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt API name)
        return width

    def dragEnterEvent(self, event):  # noqa: N802 (Qt API name)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_file = url.toLocalFile()
                if local_file and self._is_supported_file(local_file):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802 (Qt API name)
        for url in event.mimeData().urls():
            local_file = url.toLocalFile()
            if local_file and self._is_supported_file(local_file):
                self.file_selected.emit(self.raw_type, local_file)
                event.acceptProposedAction()
                return
        event.ignore()

    def resizeEvent(self, event):  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _open_file_dialog(self) -> None:
        start_dir = self._start_dir_provider()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {self.title()} File",
            start_dir,
            self._file_filter(),
        )
        if file_path:
            self.file_selected.emit(self.raw_type, file_path)

    def _is_supported_file(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        if self.raw_type == "afm":
            return suffix == ".ibw"
        if self.raw_type == "xrd":
            return suffix in (".xrdml", ".xml", ".ras", ".dat", ".txt", ".xy", ".ibw")
        if self.raw_type == "rsm":
            return suffix in (".xrdml", ".xml")
        return False

    def _drag_hint(self) -> str:
        if self.raw_type == "afm":
            return "Drag AFM .ibw here or click to load"
        if self.raw_type == "rsm":
            return "Drag RSM .xrdml/.xml here or click to load"
        return "Drag XRD scan here or click to load"

    def _file_filter(self) -> str:
        if self.raw_type == "afm":
            return "Igor Binary Wave (*.ibw);;All Files (*)"
        if self.raw_type == "rsm":
            return "RSM Files (*.xrdml *.xml);;All Files (*)"
        return "XRD Files (*.xrdml *.xml *.ras *.dat *.txt *.xy *.ibw);;All Files (*)"

    def set_result(self, file_path: str, result: RawVisualizationResult) -> None:
        """Update block after successful visualization."""
        self._loaded_file_path = file_path
        self.path_label.setText(file_path)
        if result.preview_pixmap is not None and not result.preview_pixmap.isNull():
            self._original_pixmap = result.preview_pixmap
            self.preview_label.setText("")
            self._refresh_pixmap()
            self._set_export_enabled(True)
        else:
            self._original_pixmap = None
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(result.message)
            self._set_export_enabled(False)

    def set_error(self, file_path: str, message: str) -> None:
        """Display failure message in the block."""
        self._loaded_file_path = file_path
        self.path_label.setText(file_path)
        self._original_pixmap = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(message)
        self._set_export_enabled(False)

    def _refresh_pixmap(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _set_export_enabled(self, enabled: bool) -> None:
        self.copy_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)

    def _copy_preview_to_clipboard(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            self.status_message.emit(f"{self.title()}: no image is available to copy.")
            return
        QApplication.clipboard().setPixmap(self._original_pixmap)
        self.status_message.emit(f"{self.title()}: image copied to clipboard.")

    def _export_preview_image(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            self.status_message.emit(f"{self.title()}: no image is available to export.")
            return

        if self._loaded_file_path:
            loaded_path = Path(self._loaded_file_path)
            default_path = loaded_path.with_name(f"{loaded_path.stem}_{self.raw_type}.png")
        else:
            default_path = Path(self._start_dir_provider()) / f"{self.raw_type}_preview.png"

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {self.title()} Preview",
            str(default_path),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp)",
        )
        if not output_path:
            return
        if self._original_pixmap.save(output_path):
            self.status_message.emit(f"{self.title()}: exported image to {output_path}")
        else:
            self.status_message.emit(f"{self.title()}: failed to export image.")


def visualize_raw_file(raw_type: str, file_path: str) -> RawVisualizationResult:
    """Visualize one raw file through AFM-tools/XRD-utils and return preview info."""
    try:
        if raw_type == "afm":
            backend, raw_result, detail = _visualize_afm_with_tools(file_path)
        elif raw_type == "xrd":
            backend, raw_result, detail = _visualize_xrd_with_utils(file_path)
        elif raw_type == "rsm":
            backend, raw_result, detail = _visualize_rsm_with_utils(file_path)
        else:
            raise ValueError(f"Unknown raw_type '{raw_type}'. Use 'xrd', 'rsm', or 'afm'.")
    except Exception as exc:
        hint = _dependency_troubleshooting_hint(exc)
        if hint:
            raise RuntimeError(f"{exc}\n\n{hint}") from exc
        raise

    preview = _result_to_pixmap(raw_result)
    if preview is not None:
        message = f"{detail} | Embedded preview updated."
    else:
        message = f"{detail} | Visualizer ran, but no embeddable image was returned."

    _close_figure_if_needed(raw_result)
    return RawVisualizationResult(backend=backend, preview_pixmap=preview, message=message)


def _visualize_afm_with_tools(file_path: str) -> tuple[str, Any, str]:
    """Load/analyze/visualize AFM .ibw using AFM-tools functions."""
    from matplotlib import pyplot as plt

    package_name, afm_RMS_roughness, convert_scan_setting, parse_ibw, AFMVisualizer = _import_afm_tools_symbols()

    imgs, sample_name, labels, scan_size = parse_ibw(file_path)
    if imgs.ndim != 3 or imgs.shape[2] == 0:
        raise RuntimeError("AFM-tools returned invalid image stack from this .ibw file.")

    channel_index = _pick_afm_channel(labels)
    channel_name = labels[channel_index]
    image = imgs[:, :, channel_index]
    scan_setting = convert_scan_setting(scan_size)
    rms = float(afm_RMS_roughness(image))

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    viz = AFMVisualizer(
        colorbar_setting={
            "colorbar_type": "percent",
            "colorbar_range": (2, 98),
            "outliers_std": 5,
            "symmetric_clim": False,
            "visible": True,
        },
        zero_mean=False,
        scalebar=True,
        debug=False,
    )
    viz.viz(
        img=image,
        scan_size=scan_setting,
        fig=fig,
        ax=ax,
        title=f"{sample_name} | {channel_name}",
        cbar_unit="nm",
    )
    fig.tight_layout()

    backend = f"{package_name}.afm_utils.parse_ibw + {package_name}.afm_viz.AFMVisualizer.viz"
    detail = (
        f"AFM loaded: sample={sample_name}, channel={channel_name}, "
        f"RMS={rms:.4g} m"
    )
    return backend, fig, detail


def _import_afm_tools_symbols() -> tuple[str, Callable[..., float], Callable[..., dict], Callable[..., Any], type]:
    """Import the AFM-tools APIs used by the visualizer.

    AFM-tools may import `mayavi` from package `__init__`, even though
    the 2D visualization path used here does not need it. This loader first
    tries normal imports and, if that fails due to missing `mayavi`, it bypasses
    package `__init__` and imports only required submodules.
    """
    candidate_packages = ("afm_tools", "afm_learn")
    last_error: Optional[Exception] = None

    for package_name in candidate_packages:
        try:
            afm_img = importlib.import_module(f"{package_name}.afm_image_analyzer")
            afm_utils = importlib.import_module(f"{package_name}.afm_utils")
            afm_viz = importlib.import_module(f"{package_name}.afm_viz")
            return (
                package_name,
                afm_img.afm_RMS_roughness,
                afm_utils.convert_scan_setting,
                afm_utils.parse_ibw,
                afm_viz.AFMVisualizer,
            )
        except ModuleNotFoundError as exc:
            # Optional 3D dependency; use targeted import path that bypasses package __init__.
            if exc.name == "mayavi":
                return _import_afm_tools_without_mayavi(package_name)
            # If this candidate package itself is missing, keep trying.
            if exc.name == package_name or (exc.name and exc.name.startswith(f"{package_name}.")):
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(
        "AFM-tools package is not installed. Expected one of: afm_tools, afm_learn."
    )


def _import_afm_tools_without_mayavi(
    package_name: str,
) -> tuple[str, Callable[..., float], Callable[..., dict], Callable[..., Any], type]:
    """Load AFM-tools submodules without executing package `__init__`."""
    afm_pkg_dir = _find_package_dir(package_name)
    if afm_pkg_dir is None:
        raise ModuleNotFoundError(
            "AFM-tools package is not installed. Expected one of: afm_tools, afm_learn."
        )

    # Remove partially imported package state left by a failed package import.
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType(package_name)
    pkg.__file__ = str(afm_pkg_dir / "__init__.py")
    pkg.__path__ = [str(afm_pkg_dir)]
    sys.modules[package_name] = pkg

    afm_utils = importlib.import_module(f"{package_name}.afm_utils")
    afm_viz = importlib.import_module(f"{package_name}.afm_viz")
    afm_img = importlib.import_module(f"{package_name}.afm_image_analyzer")

    return (
        package_name,
        afm_img.afm_RMS_roughness,
        afm_utils.convert_scan_setting,
        afm_utils.parse_ibw,
        afm_viz.AFMVisualizer,
    )


def _find_package_dir(package_name: str) -> Optional[Path]:
    """Locate an installed package directory from current interpreter `sys.path`."""
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / package_name
        if (candidate / "__init__.py").exists():
            return candidate
    return None


def _visualize_xrd_with_utils(file_path: str) -> tuple[str, Any, str]:
    """Load/analyze/visualize XRD scan using XRD-utils functions."""
    from matplotlib import pyplot as plt
    import numpy as np

    package_name, xrd_utils, xrd_viz, _ = _import_xrd_tools_modules()
    calculate_fwhm = xrd_utils.calculate_fwhm
    detect_peaks = xrd_utils.detect_peaks
    load_xrd_scan = xrd_utils.load_xrd_scan
    plot_xrd = xrd_viz.plot_xrd

    x, y = load_xrd_scan(file_path)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise RuntimeError("XRD-utils returned empty scan arrays.")

    prominence = max(float(np.ptp(y)) * 0.10, 1e-9)
    peaks_x, peaks_y = detect_peaks(x, y, num_peaks=1, prominence=prominence)
    peak_note = "peak not detected"
    if peaks_x:
        fwhm, *_ = calculate_fwhm(x, y, px=float(peaks_x[0]), fit_type="gaussian", viz=False)
        if fwhm is not None:
            peak_note = f"peak={float(peaks_x[0]):.4f}, FWHM={float(fwhm):.4f}"
        else:
            peak_note = f"peak={float(peaks_x[0]):.4f}, FWHM=unavailable"

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    plot_xrd(
        inputs=file_path,
        labels=[Path(file_path).stem],
        title="XRD Scan",
        yscale="log",
        diff=None,
        fig=fig,
        ax=ax,
        legend_style="legend",
        grid=True,
    )
    if peaks_x:
        ax.axvline(float(peaks_x[0]), color="tab:red", linestyle="--", linewidth=1)
        ax.scatter([float(peaks_x[0])], [float(peaks_y[0])], color="tab:red", s=12, zorder=5)
    fig.tight_layout()

    backend = (
        f"{package_name}.xrd_utils.load_xrd_scan/detect_peaks/calculate_fwhm + "
        f"{package_name}.xrd_viz.plot_xrd"
    )
    detail = f"XRD loaded: {Path(file_path).name}, {peak_note}"
    return backend, fig, detail


def _visualize_rsm_with_utils(file_path: str) -> tuple[str, Any, str]:
    """Load and visualize reciprocal space map using XRD-utils."""
    from matplotlib import pyplot as plt
    import numpy as np

    package_name, _, _, rsm_viz = _import_xrd_tools_modules()
    RSMPlotter = rsm_viz.RSMPlotter

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    plotter = RSMPlotter(
        plot_params={
            "reciprocal_space": True,
            "title": f"RSM: {Path(file_path).stem}",
            "log_scale": True,
        }
    )
    qx, qz, intensity = plotter.plot(file_path, ax=ax, ignore_yaxis=False)
    _ = qx, qz
    fig.tight_layout()

    intensity_arr = np.asarray(intensity)
    detail = f"RSM loaded: {Path(file_path).name}, grid={intensity_arr.shape}"
    backend = f"{package_name}.rsm_viz.RSMPlotter.plot"
    return backend, fig, detail


def _import_xrd_tools_modules() -> tuple[str, Any, Any, Any]:
    """Import XRD-utils modules from new or legacy package names."""
    candidate_packages = ("xrd_tools", "xrd_learn")
    last_error: Optional[Exception] = None

    for package_name in candidate_packages:
        try:
            xrd_utils = importlib.import_module(f"{package_name}.xrd_utils")
            xrd_viz = importlib.import_module(f"{package_name}.xrd_viz")
            rsm_viz = importlib.import_module(f"{package_name}.rsm_viz")
            return package_name, xrd_utils, xrd_viz, rsm_viz
        except ModuleNotFoundError as exc:
            if exc.name == package_name or (exc.name and exc.name.startswith(f"{package_name}.")):
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(
        "XRD-utils package is not installed. Expected one of: xrd_tools, xrd_learn."
    )


def _dependency_troubleshooting_hint(exc: Exception) -> str:
    """Return focused remediation hints for common binary dependency failures."""
    text = str(exc)
    lower = text.lower()

    if "glibcxx_" in lower and "not found" in lower:
        return (
            "Dependency runtime mismatch detected (missing GLIBCXX symbol).\n"
            "Try:\n"
            "1) conda activate pld\n"
            "2) conda install -n pld -c conda-forge libstdcxx-ng libgcc-ng\n"
            "3) python -m pip install --force-reinstall --no-cache-dir AFM-tools XRD-utils xrayutilities\n"
            "Also ensure VS Code uses the same `pld` interpreter."
        )

    if "numpy.dtype size changed" in lower or "binary incompatibility" in lower:
        return (
            "Binary package ABI mismatch detected (NumPy/C-extension versions differ).\n"
            "Try reinstalling numeric dependencies in one environment:\n"
            "1) conda activate pld\n"
            "2) python -m pip install --upgrade --force-reinstall --no-cache-dir "
            "numpy scipy matplotlib AFM-tools XRD-utils xrayutilities"
        )

    if "no module named 'mayavi'" in lower:
        return (
            "AFM-tools tried to import optional 3D dependency `mayavi`.\n"
            "This app can run AFM 2D visualization without it, but your environment still raised that import path.\n"
            "Try reinstalling AFM-tools and relaunching with the `pld` interpreter."
        )

    return ""


def _pick_afm_channel(labels: list[str]) -> int:
    """Select a preferred AFM channel index from label list."""
    preferred = ("Height", "ZSensor", "Amplitude", "Phase")
    for target in preferred:
        for idx, label in enumerate(labels):
            if target.lower() in str(label).lower():
                return idx
    return 0


def _close_figure_if_needed(result: Any) -> None:
    """Close matplotlib figure objects after preview extraction."""
    try:
        from matplotlib.figure import Figure
        from matplotlib import pyplot as plt
    except Exception:  # noqa: BLE001
        return
    if isinstance(result, Figure):
        plt.close(result)


def _result_to_pixmap(result: Any) -> Optional[QPixmap]:
    """Convert common visualizer return types to a Qt pixmap when possible."""
    if result is None:
        return None

    if isinstance(result, QPixmap):
        return result
    if isinstance(result, QImage):
        return QPixmap.fromImage(result)

    if isinstance(result, str):
        if os.path.isfile(result):
            pixmap = QPixmap(result)
            return pixmap if not pixmap.isNull() else None
        return None

    if isinstance(result, dict):
        for key in ("preview", "image", "img", "pixmap", "figure", "fig", "array", "data", "result"):
            if key in result:
                pixmap = _result_to_pixmap(result[key])
                if pixmap is not None:
                    return pixmap
        return None

    if isinstance(result, (list, tuple)):
        for item in result:
            pixmap = _result_to_pixmap(item)
            if pixmap is not None:
                return pixmap
        return None

    figure_pixmap = _figure_to_pixmap(result)
    if figure_pixmap is not None:
        return figure_pixmap

    array_pixmap = _array_to_pixmap(result)
    if array_pixmap is not None:
        return array_pixmap

    return None


def _figure_to_pixmap(obj: Any) -> Optional[QPixmap]:
    """Convert matplotlib figure object to pixmap."""
    try:
        from io import BytesIO
        from matplotlib.figure import Figure
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(obj, Figure):
        return None

    buffer = BytesIO()
    obj.savefig(buffer, format="png", bbox_inches="tight")
    data = buffer.getvalue()
    pixmap = QPixmap()
    loaded = pixmap.loadFromData(data, "PNG")
    if loaded and not pixmap.isNull():
        return pixmap
    return None


def _array_to_pixmap(obj: Any) -> Optional[QPixmap]:
    """Convert numpy array to pixmap (grayscale or RGB)."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(obj, np.ndarray):
        return None
    if obj.size == 0:
        return None

    arr = np.asarray(obj)
    if arr.ndim == 2:
        normalized = _normalize_to_uint8(arr)
        h, w = normalized.shape
        image = QImage(normalized.data, w, h, normalized.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(image.copy())

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        normalized = _normalize_to_uint8(arr)
        h, w, c = normalized.shape
        if c == 3:
            image = QImage(normalized.data, w, h, normalized.strides[0], QImage.Format_RGB888)
        else:
            image = QImage(normalized.data, w, h, normalized.strides[0], QImage.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())

    return None


def _normalize_to_uint8(arr):
    """Normalize array values into uint8 [0, 255]."""
    import numpy as np

    arr_float = arr.astype(float)
    low = float(arr_float.min())
    high = float(arr_float.max())
    if high <= low:
        return np.zeros(arr_float.shape, dtype=np.uint8)
    scaled = (arr_float - low) * (255.0 / (high - low))
    return scaled.clip(0, 255).astype(np.uint8)

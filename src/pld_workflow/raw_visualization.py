"""Raw XRD/AFM visualization helpers and drag/drop preview widgets."""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGroupBox, QLabel, QSizePolicy, QVBoxLayout


@dataclass(frozen=True)
class RawFileTypeSpec:
    """Configuration describing one supported raw-data input family."""

    title: str
    extensions: tuple[str, ...]
    dialog_filter: str
    placeholder_text: str
    backend_family: str

    @property
    def extension_summary(self) -> str:
        """Return a short human-readable extension list for status messages."""
        return ", ".join(self.extensions)


@dataclass
class RawVisualizationResult:
    """Result of a raw-data visualization call."""

    backend: str
    preview_pixmap: Optional[QPixmap]
    message: str


RAW_FILE_SPECS: Dict[str, RawFileTypeSpec] = {
    "xrd": RawFileTypeSpec(
        title="XRD Scan",
        extensions=(".xrdml", ".xml", ".ras", ".dat", ".txt", ".xy"),
        dialog_filter=(
            "XRD Scan Files (*.xrdml *.xml *.ras *.dat *.txt *.xy);;"
            "All Files (*)"
        ),
        placeholder_text="Drag an XRD scan file here or click to load",
        backend_family="xrd",
    ),
    "rsm": RawFileTypeSpec(
        title="RSM",
        extensions=(".xrdml", ".xml"),
        dialog_filter="RSM Files (*.xrdml *.xml);;All Files (*)",
        placeholder_text="Drag an RSM file here or click to load",
        backend_family="rsm",
    ),
    "afm": RawFileTypeSpec(
        title="AFM (.ibw)",
        extensions=(".ibw",),
        dialog_filter="Igor Binary Wave (*.ibw);;All Files (*)",
        placeholder_text="Drag an AFM .ibw file here or click to load",
        backend_family="afm",
    ),
}


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
    """Square drag/drop block used by the standalone raw-data visualizer."""

    file_selected = pyqtSignal(str, str)
    status_message = pyqtSignal(str)

    def __init__(self, raw_type: str, title: str, start_dir_provider: Callable[[], str], parent=None):
        spec = get_raw_file_spec(raw_type)
        super().__init__(title or spec.title, parent)
        self.raw_type = raw_type
        self.spec = spec
        self._start_dir_provider = start_dir_provider
        self._original_pixmap: Optional[QPixmap] = None

        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 280)

        self.preview_label = ClickablePreviewLabel(self.spec.placeholder_text)
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

        layout = QVBoxLayout()
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.path_label)
        self.setLayout(layout)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt API name)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt API name)
        return width

    def dragEnterEvent(self, event):  # noqa: N802 (Qt API name)
        if self._first_supported_url(event.mimeData().urls()) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802 (Qt API name)
        supported_file = self._first_supported_url(event.mimeData().urls())
        if supported_file is not None:
            self.file_selected.emit(self.raw_type, supported_file)
            event.acceptProposedAction()
            return

        self.status_message.emit(
            f"{self.spec.title} supports: {self.spec.extension_summary}"
        )
        event.ignore()

    def resizeEvent(self, event):  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _open_file_dialog(self) -> None:
        start_dir = self._start_dir_provider()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load {self.spec.title}",
            start_dir,
            self.spec.dialog_filter,
        )
        if not file_path:
            return
        if not self._is_supported_file(file_path):
            self.status_message.emit(
                f"{self.spec.title} supports: {self.spec.extension_summary}"
            )
            return
        self.file_selected.emit(self.raw_type, file_path)

    def set_result(self, file_path: str, result: RawVisualizationResult) -> None:
        """Update the block after a successful visualization attempt."""
        self.path_label.setText(file_path)
        if result.preview_pixmap is not None and not result.preview_pixmap.isNull():
            self._original_pixmap = result.preview_pixmap
            self.preview_label.setText("")
            self._refresh_pixmap()
            return

        self._original_pixmap = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(result.message)

    def set_error(self, file_path: str, message: str) -> None:
        """Display a failure message in the block."""
        self.path_label.setText(file_path)
        self._original_pixmap = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(message)

    def _first_supported_url(self, urls) -> Optional[str]:
        """Return the first dropped local file that matches this block type."""
        for url in urls:
            local_file = url.toLocalFile()
            if local_file and self._is_supported_file(local_file):
                return local_file
        return None

    def _is_supported_file(self, file_path: str) -> bool:
        """Return True when the file suffix matches the configured raw type."""
        _, extension = os.path.splitext(file_path)
        return extension.lower() in self.spec.extensions

    def _refresh_pixmap(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)


def get_raw_file_spec(raw_type: str) -> RawFileTypeSpec:
    """Return the file-handling spec for the requested raw-data family."""
    try:
        return RAW_FILE_SPECS[raw_type]
    except KeyError as exc:
        supported = ", ".join(sorted(RAW_FILE_SPECS))
        raise ValueError(f"Unknown raw_type '{raw_type}'. Use one of: {supported}.") from exc


def visualize_raw_file(raw_type: str, file_path: str) -> RawVisualizationResult:
    """Run the configured raw-data backend and attempt to build an embedded preview."""
    called_backend, result = _run_learn_visualizer(raw_type, file_path)
    preview = _result_to_pixmap(result)
    if preview is not None:
        message = "Embedded preview updated."
    else:
        message = "Visualizer launched (no embeddable image returned)."
    return RawVisualizationResult(backend=called_backend, preview_pixmap=preview, message=message)


def _call_with_file_path(func: Callable[..., Any], file_path: str) -> Any:
    """Call a visualizer function using common file-path argument conventions."""
    try:
        return func(file_path)
    except TypeError:
        signature = inspect.signature(func)
        for param_name in ("file_path", "path", "filename", "file", "ibw_file"):
            if param_name in signature.parameters:
                return func(**{param_name: file_path})
        raise


def _run_learn_visualizer(raw_type: str, file_path: str) -> tuple[str, Any]:
    """Execute visualization from XRD/AFM helper packages when available."""
    spec = get_raw_file_spec(raw_type)
    known_backend_result = _run_known_backend(raw_type, file_path)
    if known_backend_result is not None:
        return known_backend_result

    module_candidates: Dict[str, list[str]] = {
        "xrd": ["xrd_learn", "xrdlearn", "XRD_Learn", "xrd_utils", "xrdutils"],
        "rsm": ["xrd_learn", "xrdlearn", "XRD_Learn", "xrd_utils", "xrdutils"],
        "afm": ["afm_learn", "afmlearn", "AFM_Learn", "afm_tools", "afmtools"],
    }
    function_candidates: Dict[str, list[str]] = {
        "xrd": [
            "visualize_xrd",
            "plot_xrd",
            "visualize_scan",
            "plot_scan",
            "visualize_ibw",
            "visualize",
            "plot",
            "main",
        ],
        "rsm": [
            "visualize_rsm",
            "plot_rsm",
            "visualize_xrd",
            "plot_xrd",
            "visualize",
            "plot",
            "main",
        ],
        "afm": [
            "visualize_afm",
            "plot_afm",
            "visualize_ibw",
            "plot_ibw",
            "visualize",
            "plot",
            "main",
        ],
    }

    loaded_module = None
    for module_name in module_candidates[raw_type]:
        try:
            loaded_module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if loaded_module is None:
        package_hint = {
            "xrd": "an XRD visualization backend such as XRD-Learn/XRD-utils",
            "rsm": "an RSM/XRD visualization backend such as XRD-Learn/XRD-utils",
            "afm": "an AFM visualization backend such as AFM-Learn/AFM-tools",
        }[spec.backend_family]
        raise RuntimeError(
            f"No backend is installed for {spec.title}. Install {package_hint}, then retry."
        )

    for function_name in function_candidates[raw_type]:
        candidate = getattr(loaded_module, function_name, None)
        if callable(candidate):
            result = _call_with_file_path(candidate, file_path)
            return f"{loaded_module.__name__}.{function_name}", result

    expected_names = ", ".join(function_candidates[raw_type])
    raise RuntimeError(
        f"No compatible visualize function was found in module '{loaded_module.__name__}'. "
        f"Expected one of: {expected_names}"
    )


def _run_known_backend(raw_type: str, file_path: str) -> tuple[str, Any] | None:
    """Use direct adapters for the published AFM-tools and XRD-utils packages."""
    if raw_type == "xrd":
        return _run_xrd_utils_scan(file_path)
    if raw_type == "rsm":
        return _run_xrd_utils_rsm(file_path)
    if raw_type == "afm":
        return _run_afm_tools_ibw(file_path)
    return None


def _run_xrd_utils_scan(file_path: str) -> tuple[str, Any] | None:
    """Render a standard XRD scan using `XRD-utils` when it is installed."""
    try:
        from matplotlib import pyplot as plt
        from xrd_utils.xrd_viz import plot_xrd
    except Exception:  # noqa: BLE001
        return None

    figure, axis = plt.subplots(figsize=(6, 4))
    plot_xrd([file_path], [os.path.basename(file_path)], fig=figure, ax=axis, diff=None, yscale="log")
    figure.tight_layout()
    return "xrd_utils.xrd_viz.plot_xrd", figure


def _run_xrd_utils_rsm(file_path: str) -> tuple[str, Any] | None:
    """Render an RSM map using `XRD-utils` when it is installed."""
    try:
        from matplotlib import pyplot as plt
        from xrd_utils.rsm_viz import RSMPlotter
    except Exception:  # noqa: BLE001
        return None

    figure, axis = plt.subplots(figsize=(6, 5))
    plotter = RSMPlotter()
    plotter.plot(file_path, ax=axis)
    figure.tight_layout()
    return "xrd_utils.rsm_viz.RSMPlotter.plot", figure


def _run_afm_tools_ibw(file_path: str) -> tuple[str, Any] | None:
    """Render an AFM image using `AFM-tools` when it is installed."""
    try:
        import numpy as np
        from afm_tools.afm_utils import parse_ibw
        from afm_tools.afm_viz import AFMVisualizer
    except Exception:  # noqa: BLE001
        return None

    images, sample_name, labels, scan_size = parse_ibw(file_path)
    if images.ndim != 3 or images.shape[2] == 0:
        raise RuntimeError("AFM-tools parsed the file, but no image channels were returned.")

    channel_index = 0
    preferred_labels = ("Height", "ZSensor", "Amplitude", "Phase")
    for preferred_label in preferred_labels:
        if preferred_label in labels:
            channel_index = labels.index(preferred_label)
            break

    image = np.asarray(images[:, :, channel_index])
    visualizer = AFMVisualizer(
        colorbar_setting={
            "colorbar_type": "percent",
            "colorbar_range": (0.2, 99.8),
            "outliers_std": 5,
            "symmetric_clim": False,
            "visible": True,
        },
        zero_mean=False,
        scalebar=True,
        debug=False,
    )
    figure, _axis = visualizer.viz(
        img=image,
        scan_size=scan_size,
        title=f"{sample_name} - {labels[channel_index]}",
    )
    figure.tight_layout()
    return "afm_tools.afm_viz.AFMVisualizer.viz", figure


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
    """Convert a matplotlib figure object into a pixmap."""
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
    """Convert a numpy array into a pixmap when the shape is image-like."""
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
        height, width = normalized.shape
        image = QImage(normalized.data, width, height, normalized.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(image.copy())

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        normalized = _normalize_to_uint8(arr)
        height, width, channels = normalized.shape
        if channels == 3:
            image = QImage(normalized.data, width, height, normalized.strides[0], QImage.Format_RGB888)
        else:
            image = QImage(normalized.data, width, height, normalized.strides[0], QImage.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())

    return None


def _normalize_to_uint8(arr):
    """Normalize numeric array values into unsigned 8-bit image data."""
    import numpy as np

    arr_float = arr.astype(float)
    low = float(arr_float.min())
    high = float(arr_float.max())
    if high <= low:
        return np.zeros(arr_float.shape, dtype=np.uint8)
    scaled = (arr_float - low) * (255.0 / (high - low))
    return scaled.clip(0, 255).astype(np.uint8)

"""Application entrypoint for XRD/AFM raw-data visualizer."""

from __future__ import annotations

import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QVBoxLayout, QWidget

from .raw_visualization import RawDataDropBlock, visualize_raw_file


class RawVisualizerWindow(QWidget):
    """Standalone window for XRD/AFM drag/drop visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD Raw Data Visualizer")
        self.setMinimumSize(1240, 460)

        self._last_dir = os.getcwd()
        self.status_label = QLabel(
            "AFM expects .ibw. XRD scan accepts xrdml/xml/ras/dat/txt/xy. RSM expects xrdml/xml."
        )
        self.status_label.setWordWrap(True)

        self.xrd_block = RawDataDropBlock("xrd", "XRD Scan", self._start_dir)
        self.rsm_block = RawDataDropBlock("rsm", "RSM", self._start_dir)
        self.afm_block = RawDataDropBlock("afm", "AFM (.ibw)", self._start_dir)
        self.xrd_block.file_selected.connect(self._on_raw_file_selected)
        self.rsm_block.file_selected.connect(self._on_raw_file_selected)
        self.afm_block.file_selected.connect(self._on_raw_file_selected)
        self.xrd_block.status_message.connect(self._set_status)
        self.rsm_block.status_message.connect(self._set_status)
        self.afm_block.status_message.connect(self._set_status)

        block_row = QHBoxLayout()
        block_row.addWidget(self.xrd_block)
        block_row.addWidget(self.rsm_block)
        block_row.addWidget(self.afm_block)

        layout = QVBoxLayout()
        layout.addLayout(block_row, 1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def _start_dir(self) -> str:
        return self._last_dir

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_raw_file_selected(self, raw_type: str, file_path: str) -> None:
        self._last_dir = os.path.dirname(file_path) or self._last_dir
        label_map = {"xrd": "XRD", "rsm": "RSM", "afm": "AFM"}
        block_map = {"xrd": self.xrd_block, "rsm": self.rsm_block, "afm": self.afm_block}
        label = label_map.get(raw_type, raw_type.upper())
        block = block_map.get(raw_type)
        if block is None:
            self._set_status(f"Unsupported type: {raw_type}")
            return

        try:
            result = visualize_raw_file(raw_type, file_path)
        except Exception as exc:  # noqa: BLE001
            message = f"{label} visualization failed: {exc}"
            block.set_error(file_path, message)
            self._set_status(message)
            return

        block.set_result(file_path, result)
        self._set_status(f"{label}: {result.message} ({result.backend})")


def main() -> int:
    """Launch the raw-data visualizer window."""
    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)
    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QLabel")

    window = RawVisualizerWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

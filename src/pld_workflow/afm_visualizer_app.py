"""Application entrypoint for the standalone AFM/PFM raw-data visualizer."""

from __future__ import annotations

import os
import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from .raw_visualization import AfmDataDropBlock


class AfmVisualizerWindow(QWidget):
    """Standalone window for AFM/PFM drag/drop visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLD AFM / PFM Visualizer")
        self.setMinimumSize(960, 760)

        self._last_dir = os.getcwd()
        self.status_label = QLabel(
            "AFM/PFM expects .ibw files. "
            "Select one or more channels to preview. Height is selected by default."
        )
        self.status_label.setWordWrap(True)

        self.afm_block = AfmDataDropBlock(self._start_dir)
        self.afm_block.status_message.connect(self._set_status)
        self.afm_block.file_loaded.connect(self._on_afm_file_loaded)

        layout = QVBoxLayout()
        layout.addWidget(self.afm_block, 1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def _start_dir(self) -> str:
        return self._last_dir

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_afm_file_loaded(self, file_path: str) -> None:
        self._last_dir = os.path.dirname(file_path) or self._last_dir


def main() -> int:
    """Launch the AFM/PFM visualizer window."""

    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)
    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QLabel")

    window = AfmVisualizerWindow()
    window.show()
    return app.exec_()


__all__ = ["AfmVisualizerWindow", "main"]

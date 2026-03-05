"""Application entrypoint for the PLD parameter form."""

from __future__ import annotations

import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from .form import GenerateForm


def main() -> int:
    """Launch the desktop form UI.

    Returns
    -------
    int
        Qt event loop return code.
    """
    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)

    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QComboBox")
    app.setFont(custom_font, "QLabel")
    app.setFont(custom_font, "QLineEdit")
    app.setFont(custom_font, "QPlainTextEdit")

    window = GenerateForm(version="parameter")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

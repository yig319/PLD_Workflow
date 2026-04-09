"""Backward-compatible wrapper for the XRD visualizer app.

The raw visualizer is now split into separate diffraction and AFM/PFM apps.
This module keeps the historic import path working and points it to the
XRD visualizer.
"""

from .xrd_visualizer_app import XrdVisualizerWindow, main

XrdRsmVisualizerWindow = XrdVisualizerWindow

__all__ = ["XrdVisualizerWindow", "XrdRsmVisualizerWindow", "main"]

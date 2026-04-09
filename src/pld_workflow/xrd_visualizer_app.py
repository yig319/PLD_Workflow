"""Canonical app entrypoint for the XRD-family visualizer."""

from .xrd_rsm_visualizer_app import XrdRsmVisualizerWindow, XrdVisualizerWindow, main

__all__ = ["XrdVisualizerWindow", "XrdRsmVisualizerWindow", "main"]

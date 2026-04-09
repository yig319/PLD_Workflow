"""Dedicated XRD and RSM preview helpers for the raw-data visualizer."""

from __future__ import annotations

import os
from typing import Any


def render_xrd_preview(file_path: str) -> tuple[str, Any] | None:
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


def render_rsm_preview(file_path: str) -> tuple[str, Any] | None:
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


__all__ = ["render_rsm_preview", "render_xrd_preview"]

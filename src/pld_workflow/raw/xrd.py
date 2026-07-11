"""XRD and RSM preview compatibility wrappers around XRD-utils."""

from __future__ import annotations

import os
from typing import Any


def render_xrd_preview(file_path: str) -> tuple[str, Any] | None:
    """Render a standard XRD scan using XRD-utils when it is installed."""

    try:
        from xrd_utils.xrd_viz import render_xrd_preview as xrd_render_xrd_preview
    except Exception:  # noqa: BLE001
        return _legacy_render_xrd_preview(file_path)

    return xrd_render_xrd_preview(file_path)


def render_rsm_preview(file_path: str) -> tuple[str, Any] | None:
    """Render an RSM map using XRD-utils when it is installed."""

    try:
        from xrd_utils.rsm_viz import render_rsm_preview as xrd_render_rsm_preview
    except Exception:  # noqa: BLE001
        return _legacy_render_rsm_preview(file_path)

    return xrd_render_rsm_preview(file_path)


def _legacy_render_xrd_preview(file_path: str) -> tuple[str, Any] | None:
    try:
        from matplotlib import pyplot as plt
        from xrd_utils.xrd_viz import plot_xrd
    except Exception:  # noqa: BLE001
        return None

    figure, axis = plt.subplots(figsize=(6, 4))
    plot_xrd(_xrd_plot_input(file_path), [os.path.basename(file_path)], fig=figure, ax=axis, diff=None, yscale="log")
    figure.tight_layout()
    return "xrd_utils.xrd_viz.plot_xrd", figure


def _legacy_render_rsm_preview(file_path: str) -> tuple[str, Any] | None:
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


def _xrd_plot_input(file_path: str):
    try:
        from xrd_utils.xrd_utils import load_xrd_scans
    except Exception:  # noqa: BLE001
        return [file_path]
    return load_xrd_scans([file_path])


__all__ = ["render_rsm_preview", "render_xrd_preview"]

"""AFM and PFM preview helpers for the raw-data visualizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AfmDataset:
    """Parsed AFM/PFM data cached for re-rendering in the preview widget."""

    file_path: str
    images: np.ndarray
    sample_name: str
    labels: list[str]
    scan_size: object


@dataclass(slots=True)
class AfmPreviewOptions:
    """Interactive settings that control AFM preview rendering."""

    selected_channel_indices: list[int]
    show_metric_overlay: bool = False


@dataclass(slots=True)
class AfmPreviewRender:
    """Rendered AFM preview plus a short UI status message."""

    figure: object
    message: str


def load_afm_dataset(file_path: str) -> AfmDataset:
    """Parse one IBW file and return the image stack plus channel labels."""

    try:
        from afm_tools.afm_utils import parse_ibw
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("AFM-tools is required to preview AFM/PFM IBW files.") from exc

    images, sample_name, labels, scan_size = parse_ibw(file_path)
    if images.ndim != 3 or images.shape[2] == 0:
        raise RuntimeError("AFM-tools parsed the file, but no image channels were returned.")

    return AfmDataset(
        file_path=file_path,
        images=np.asarray(images),
        sample_name=str(sample_name),
        labels=[str(label) for label in labels],
        scan_size=scan_size,
    )


def preferred_channel_index(labels: list[str]) -> int:
    """Choose a sensible default channel for the initial preview."""

    preferred_labels = ("Height", "ZSensor", "Amplitude", "Phase")
    for preferred_label in preferred_labels:
        if preferred_label in labels:
            return labels.index(preferred_label)
    return 0


def render_afm_preview(dataset: AfmDataset, options: AfmPreviewOptions) -> AfmPreviewRender:
    """Render one AFM/PFM preview figure from cached channel data."""

    try:
        import matplotlib.pyplot as plt
        from afm_tools.afm_viz import AFMVisualizer, convert_with_unit
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("AFM-tools with matplotlib is required to render AFM previews.") from exc

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

    channel_indices = _normalize_selected_indices(options.selected_channel_indices, len(dataset.labels))
    if not channel_indices:
        channel_indices = [preferred_channel_index(dataset.labels)]

    if len(channel_indices) > 1:
        n_channels = len(channel_indices)
        n_cols = 2 if n_channels <= 4 else 3
        n_rows = int(math.ceil(n_channels / float(n_cols)))
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
        axes_flat = np.atleast_1d(axes).ravel()

        for axis, channel_index in zip(axes_flat, channel_indices):
            channel_label = dataset.labels[channel_index]
            image = np.asarray(dataset.images[:, :, channel_index], dtype=float)
            metric_text, colorbar_unit = _describe_afm_metric(
                channel_label,
                image,
                convert_with_unit=convert_with_unit,
            )
            visualizer.viz(
                img=image,
                scan_size=dataset.scan_size,
                fig=figure,
                ax=axis,
                title=channel_label,
                cbar_unit=colorbar_unit,
            )
            if options.show_metric_overlay and _should_show_metric_overlay(channel_label, multiple_plots=True):
                _add_metric_overlay(axis, f"RMS = {metric_text}")

        for axis in axes_flat[n_channels:]:
            axis.set_visible(False)

        figure.suptitle(dataset.sample_name, fontsize=12)
        figure.subplots_adjust(left=0.035, right=0.985, bottom=0.05, top=0.90, wspace=0.02, hspace=0.10)
        return AfmPreviewRender(
            figure=figure,
            message=f"AFM preview updated with {n_channels} selected channels.",
        )

    selected_index = channel_indices[0]
    channel_label = dataset.labels[selected_index]
    image = np.asarray(dataset.images[:, :, selected_index], dtype=float)
    metric_text, colorbar_unit = _describe_afm_metric(
        channel_label,
        image,
        convert_with_unit=convert_with_unit,
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    visualizer.viz(
        img=image,
        scan_size=dataset.scan_size,
        fig=figure,
        ax=axis,
        title=None,
        cbar_unit=colorbar_unit,
    )
    figure.suptitle(f"{dataset.sample_name} - {channel_label}", fontsize=12)
    if options.show_metric_overlay:
        _add_metric_overlay(axis, f"RMS = {metric_text}")
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.90)
    return AfmPreviewRender(
        figure=figure,
        message=f"AFM preview updated for {channel_label} with RMS = {metric_text}.",
    )


def _add_metric_overlay(axis, metric_text: str) -> None:
    """Draw a small metric label directly on top of one preview image."""

    axis.text(
        0.02,
        0.98,
        metric_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3.0},
    )


def _compute_rms_metric(image: np.ndarray) -> float:
    """Compute the root-mean-square value on the finite pixels of one channel."""

    values = np.asarray(image, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    centered = finite_values - float(finite_values.mean())
    return float(np.sqrt(np.mean(centered**2)))


def _should_show_metric_overlay(channel_label: str, *, multiple_plots: bool) -> bool:
    """Return whether the metric overlay should be drawn for one channel."""

    if not multiple_plots:
        return True
    return channel_label.strip().lower() == "height"


def _normalize_selected_indices(selected_channel_indices: list[int], channel_count: int) -> list[int]:
    """Return the selected channel indices as an in-range deduplicated list."""

    normalized: list[int] = []
    seen: set[int] = set()
    for index in selected_channel_indices:
        if index < 0 or index >= channel_count:
            continue
        if index in seen:
            continue
        normalized.append(int(index))
        seen.add(int(index))
    return normalized


def _describe_afm_metric(
    channel_label: str,
    image: np.ndarray,
    *,
    convert_with_unit,
) -> tuple[str, str]:
    """Return formatted RMS text and colorbar unit for one channel."""

    rms_value = _compute_rms_metric(image)
    normalized = channel_label.strip().lower()

    if normalized in {"phase", "latphase"}:
        return f"{rms_value:.2f} deg", "deg"

    metric_text = convert_with_unit(rms_value)
    unit = metric_text.split()[-1] if " " in metric_text else "nm"
    return metric_text, unit


__all__ = [
    "AfmDataset",
    "AfmPreviewOptions",
    "AfmPreviewRender",
    "load_afm_dataset",
    "preferred_channel_index",
    "render_afm_preview",
]

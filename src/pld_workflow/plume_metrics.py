"""Helpers for basic plume-shape metrics used in notebooks and analysis demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


METRIC_NAMES = [
    "area",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "centroid-0",
    "centroid-1",
    "orientation",
    "eccentricity",
    "perimeter",
    "distance",
    "velocity",
]


@dataclass
class PlumeMetrics:
    """Compute simple geometric plume metrics from packed plume frames.

    Parameters
    ----------
    plumes:
        Array shaped like `(plume_index, frame_index, height, width)`.
    condition:
        Free-text label describing the experimental condition for this batch.
    """

    plumes: object
    condition: str

    def crop_clip_image(self, image, x_start: int = 50, intensity: int = 200):
        """Crop the left edge and threshold intensities for a quick plume mask."""
        np = _import_numpy()

        clipped = np.copy(image)[:, x_start:]
        clipped[clipped < intensity] = 0
        clipped[clipped > intensity] = intensity
        return clipped

    def get_metrics(self):
        """Return the legacy stacked metric array used by existing notebooks."""
        np = _import_numpy()
        pd = _import_pandas()
        regionprops_table = _import_regionprops_table()

        metric_lists: dict[str, list[list[float]]] = {
            name: [] for name in METRIC_NAMES[:-2]
        }

        for plume_frames in self.plumes:
            for name in metric_lists:
                metric_lists[name].append([])

            for frame in plume_frames:
                processed = self.crop_clip_image(frame, x_start=50, intensity=200)
                if np.sum(processed) == 0:
                    for name in metric_lists:
                        metric_lists[name][-1].append(0.0)
                    continue

                props = regionprops_table(
                    processed,
                    properties=(
                        "area",
                        "area_filled",
                        "axis_major_length",
                        "axis_minor_length",
                        "centroid",
                        "orientation",
                        "eccentricity",
                        "perimeter",
                    ),
                )
                data = pd.DataFrame(props)
                for name in metric_lists:
                    metric_lists[name][-1].append(float(data[name].iloc[0]))

        stacked_metrics = [np.stack(metric_lists[name]) for name in METRIC_NAMES[:-2]]
        distance, velocity = self.calculate_speed()
        stacked_metrics.extend([distance, velocity])
        return np.stack(stacked_metrics)

    def calculate_speed(self):
        """Estimate plume-front distance and frame-to-frame velocity."""
        np = _import_numpy()

        distances = []
        for plume_frames in self.plumes:
            plume_distance = []
            for frame_index, frame in enumerate(plume_frames):
                column_profile = np.mean(frame, axis=0)
                bright_columns = np.where(column_profile > 100)[0]
                if bright_columns.size > 0:
                    plume_distance.append(int(np.max(bright_columns)))
                elif frame_index > 0:
                    plume_distance.append(plume_distance[frame_index - 1])
                else:
                    plume_distance.append(0)
            distances.append(np.asarray(plume_distance))

        distance = np.stack(distances, axis=0)
        velocity = distance[:, 1:] - distance[:, :-1]
        velocity = np.concatenate((distance[:, :1], velocity), axis=1)
        return distance, velocity

    def to_df(self, plots_all):
        """Convert the legacy stacked metric array into a pandas DataFrame."""
        np = _import_numpy()
        pd = _import_pandas()

        metric_name_index = np.repeat(METRIC_NAMES, plots_all.shape[1] * plots_all.shape[2])
        growth_index = list(np.repeat(np.arange(plots_all.shape[1]), plots_all.shape[2])) * plots_all.shape[0]
        time_index = np.array(list(np.arange(plots_all.shape[2])) * plots_all.shape[1] * plots_all.shape[0])
        condition_list = [self.condition] * len(time_index)

        frame = pd.DataFrame(
            {
                "condition": condition_list,
                "metric": metric_name_index,
                "growth_index": growth_index,
                "time_step": time_index,
                "a.u.": plots_all.reshape(-1),
            }
        )
        frame["growth_index"] = frame["growth_index"].astype(int)
        frame["time_step"] = frame["time_step"].astype(int)
        frame["a.u."] = frame["a.u."].astype(float)
        return frame


def plot_metrics(frame, sort_by: str = "condition"):
    """Plot line traces for each plume metric using seaborn."""
    np = _import_numpy()
    plt = _import_matplotlib_pyplot()
    sns = _import_seaborn()

    plot_frame = frame.copy()
    for metric_name in METRIC_NAMES:
        sns.set(rc={"figure.figsize": (12, 8)})
        sns.set_style("white")

        if sort_by == "growth_index" and plot_frame["growth_index"].nunique() > 10:
            grouped = np.array_split(sorted(plot_frame["growth_index"].unique()), 10)
            for group in grouped:
                replacement = int(group[0])
                for index in group:
                    plot_frame.loc[plot_frame["growth_index"] == index, "growth_index"] = replacement

        sns.lineplot(
            data=plot_frame.loc[plot_frame["metric"] == metric_name],
            x="time_step",
            y="a.u.",
            hue=sort_by,
        )
        plt.title(metric_name)
        plt.show()

    return plot_frame


def plot_metrics_heatmap(frame, frame_range: tuple[int, int]):
    """Plot one heatmap per plume metric across growth index and time."""
    plt = _import_matplotlib_pyplot()
    sns = _import_seaborn()

    for metric_name in frame.metric.unique():
        metric_frame = frame.loc[frame["metric"] == metric_name]
        metric_frame = metric_frame.loc[metric_frame["time_step"] < frame_range[1]]
        metric_frame = metric_frame.loc[metric_frame["time_step"] > frame_range[0]]
        pivot = metric_frame.pivot(index="growth_index", columns="time_step", values="a.u.")
        sns.heatmap(pivot).set(title=metric_name)
        plt.show()


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for plume metric plotting.") from exc
    return plt


def _import_numpy():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required for plume metric analysis.") from exc
    return np


def _import_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required for plume metric analysis.") from exc
    return pd


def _import_regionprops_table():
    try:
        from skimage.measure import regionprops_table
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scikit-image is required for plume metric analysis.") from exc
    return regionprops_table


def _import_seaborn():
    try:
        import seaborn as sns
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("seaborn is required for plume metric plotting.") from exc
    return sns

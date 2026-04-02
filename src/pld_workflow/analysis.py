"""Independent utilities for analyzing trends across PLD JSON records."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def discover_json_records(paths: Iterable[str | Path], recursive: bool = True) -> list[Path]:
    """Collect JSON record files from a mix of file and directory inputs."""
    record_paths: list[Path] = []
    seen: set[Path] = set()

    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".json":
            if path not in seen:
                record_paths.append(path)
                seen.add(path)
            continue

        if path.is_dir():
            iterator = path.rglob("*.json") if recursive else path.glob("*.json")
            for record_path in sorted(iterator):
                if record_path not in seen:
                    record_paths.append(record_path)
                    seen.add(record_path)

    return sorted(record_paths)


def load_record_dataframe(paths: Iterable[str | Path], recursive: bool = True):
    """Load many JSON records into one long-form pandas DataFrame."""
    pd = _import_pandas()
    rows: list[dict[str, Any]] = []
    for record_path in discover_json_records(paths, recursive=recursive):
        rows.extend(_record_to_rows(record_path))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if "timestamp" in frame.columns:
        frame = frame.sort_values(
            by=["timestamp", "growth_id", "section", "parameter", "record_path"],
            na_position="last",
        ).reset_index(drop=True)
    return frame


def list_available_parameters(paths: Iterable[str | Path], recursive: bool = True):
    """Return unique section/parameter combinations found across JSON records."""
    pd = _import_pandas()
    frame = load_record_dataframe(paths, recursive=recursive)
    if frame.empty:
        return pd.DataFrame(columns=["section", "parameter", "count"])

    summary = (
        frame.groupby(["section", "parameter"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["section", "parameter"])
        .reset_index(drop=True)
    )
    return summary


def build_parameter_trend(
    paths: Iterable[str | Path],
    parameter: str,
    section: str | None = None,
    recursive: bool = True,
):
    """Return a DataFrame describing one parameter's trend over time."""
    frame = load_record_dataframe(paths, recursive=recursive)
    if frame.empty:
        return frame

    trend = frame.loc[frame["parameter"] == parameter].copy()
    if section is not None:
        trend = trend.loc[trend["section"] == section].copy()

    if trend.empty:
        return trend

    trend = trend.sort_values(by=["timestamp", "record_path"], na_position="last").reset_index(drop=True)
    return trend


def plot_parameter_trend(trend_frame, value_column: str = "numeric_value", ax=None):
    """Plot one parameter trend using matplotlib and return the axes object."""
    plt = _import_matplotlib_pyplot()

    if trend_frame.empty:
        raise ValueError("No records matched the requested parameter.")

    if value_column not in trend_frame.columns:
        raise ValueError(f"Column '{value_column}' is not present in the trend DataFrame.")

    plot_frame = trend_frame.dropna(subset=[value_column]).copy()
    if plot_frame.empty:
        raise ValueError("The selected parameter does not contain numeric values to plot.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    x_values = plot_frame["timestamp"].where(plot_frame["timestamp"].notna(), plot_frame["record_name"])
    ax.plot(x_values, plot_frame[value_column], marker="o", linewidth=1.5)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(plot_frame["parameter"].iloc[0])
    section_label = plot_frame["section"].iloc[0] if plot_frame["section"].nunique() == 1 else "multiple sections"
    ax.set_title(f"{plot_frame['parameter'].iloc[0]} trend ({section_label})")
    ax.grid(True, alpha=0.3)

    for index, row in plot_frame.iterrows():
        label = row.get("growth_id") or row.get("record_name") or ""
        if label:
            x_value = x_values.loc[index]
            ax.annotate(label, (x_value, row[value_column]), textcoords="offset points", xytext=(0, 6), ha="center")

    plt.gcf().autofmt_xdate()
    return ax


def _record_to_rows(record_path: Path) -> list[dict[str, Any]]:
    """Normalize one nested JSON record into long-form analysis rows."""
    with open(record_path, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    header = record.get("header", {}) if isinstance(record, dict) else {}
    timestamp = _parse_timestamp(header if isinstance(header, dict) else {})
    growth_id = str(header.get("Growth ID", "")) if isinstance(header, dict) else ""
    user_name = str(header.get("User Name", "")) if isinstance(header, dict) else ""
    rows: list[dict[str, Any]] = []

    if not isinstance(record, dict):
        return rows

    for section_name, section_data in record.items():
        if not isinstance(section_data, dict):
            continue

        target_index = _target_index_from_section(section_name)
        for parameter_name, value in section_data.items():
            rows.append(
                {
                    "record_path": str(record_path),
                    "record_name": record_path.name,
                    "growth_id": growth_id,
                    "user_name": user_name,
                    "timestamp": timestamp,
                    "section": section_name,
                    "target_index": target_index,
                    "parameter": str(parameter_name),
                    "value": value,
                    "numeric_value": _coerce_numeric_value(value),
                }
            )

    return rows


def _parse_timestamp(header: dict[str, Any]) -> datetime | None:
    """Parse a timestamp from the standard PLD header block when possible."""
    date_text = str(header.get("Date", "")).strip()
    time_text = str(header.get("time", header.get("Time", ""))).strip()

    if not date_text and not time_text:
        return None

    candidates = []
    combined = f"{date_text} {time_text}".strip()
    if combined:
        candidates.extend(
            [
                (combined, "%m/%d/%Y %H:%M:%S"),
                (combined, "%m/%d/%Y %H:%M"),
                (combined, "%Y-%m-%d %H:%M:%S"),
                (combined, "%Y-%m-%d %H:%M"),
            ]
        )
    if date_text:
        candidates.extend(
            [
                (date_text, "%m/%d/%Y"),
                (date_text, "%Y-%m-%d"),
                (date_text, "%m%d%Y"),
            ]
        )

    for value, fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _target_index_from_section(section_name: str) -> int | None:
    """Extract the target number from section names like `target_2`."""
    match = re.fullmatch(r"target_(\d+)", section_name.lower())
    if not match:
        return None
    return int(match.group(1))


def _coerce_numeric_value(value: Any) -> float | None:
    """Convert numeric-like values into floats for trend analysis."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for plotting parameter trends.") from exc
    return plt


def _import_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required for parameter trend analysis.") from exc
    return pd

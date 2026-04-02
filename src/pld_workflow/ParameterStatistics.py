"""Backward-compatible exports for independent JSON record analysis."""

from __future__ import annotations

from .analysis import (
    build_parameter_trend,
    discover_json_records,
    list_available_parameters,
    load_record_dataframe,
    plot_parameter_trend,
)


def PlotParameter(paths, parameter, section="target_1"):
    """Legacy helper that builds and plots one parameter trend."""
    trend = build_parameter_trend(paths, parameter=parameter, section=section)
    if not trend.empty:
        plot_parameter_trend(trend)
    return trend


__all__ = [
    "PlotParameter",
    "build_parameter_trend",
    "discover_json_records",
    "list_available_parameters",
    "load_record_dataframe",
    "plot_parameter_trend",
]

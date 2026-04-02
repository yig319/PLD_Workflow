"""Backward-compatible wrapper for `pld_workflow.analysis` helpers."""

from pld_workflow.ParameterStatistics import (
    PlotParameter,
    build_parameter_trend,
    discover_json_records,
    list_available_parameters,
    load_record_dataframe,
    plot_parameter_trend,
)

__all__ = [
    "PlotParameter",
    "build_parameter_trend",
    "discover_json_records",
    "list_available_parameters",
    "load_record_dataframe",
    "plot_parameter_trend",
]

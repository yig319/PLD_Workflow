"""Raw-data visualization adapters and Qt preview widgets."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "AfmDataDropBlock": ("pld_workflow.raw.widgets", "AfmDataDropBlock"),
    "RawDataDropBlock": ("pld_workflow.raw.widgets", "RawDataDropBlock"),
    "RawFileTypeSpec": ("pld_workflow.raw.widgets", "RawFileTypeSpec"),
    "RawVisualizationResult": ("pld_workflow.raw.widgets", "RawVisualizationResult"),
    "get_raw_file_spec": ("pld_workflow.raw.widgets", "get_raw_file_spec"),
    "load_afm_dataset": ("pld_workflow.raw.afm", "load_afm_dataset"),
    "preferred_channel_index": ("pld_workflow.raw.afm", "preferred_channel_index"),
    "render_afm_preview": ("pld_workflow.raw.afm", "render_afm_preview"),
    "render_rsm_preview": ("pld_workflow.raw.xrd", "render_rsm_preview"),
    "render_xrd_preview": ("pld_workflow.raw.xrd", "render_xrd_preview"),
    "visualize_raw_file": ("pld_workflow.raw.widgets", "visualize_raw_file"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Lazily import raw-visualization exports on first access."""

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable autocomplete results for lazy exports."""

    return sorted(set(globals()) | set(__all__))

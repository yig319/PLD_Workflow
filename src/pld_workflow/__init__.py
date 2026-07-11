"""PLD workflow desktop apps plus optional plume and analysis helpers."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "AfmVisualizerWindow": ("pld_workflow.apps.afm_pfm_visualizer", "AfmVisualizerWindow"),
    "AfmPfmVisualizerWindow": ("pld_workflow.apps.afm_pfm_visualizer", "AfmPfmVisualizerWindow"),
    "GenerateForm": ("pld_workflow.form", "GenerateForm"),
    "MessageWindow": ("pld_workflow.form", "MessageWindow"),
    "PlumeManagerWindow": ("pld_workflow.apps.plume_manager", "PlumeManagerWindow"),
    "RheedImmVisualizerWindow": ("pld_workflow.apps.rheed_imm_visualizer", "RheedImmVisualizerWindow"),
    "XrdVisualizerWindow": ("pld_workflow.apps.xrd_visualizer", "XrdVisualizerWindow"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Lazily import public package-level exports on first access."""

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

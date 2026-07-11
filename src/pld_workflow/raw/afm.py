"""Small AFM/PFM preview adapter around AFM-tools."""

from __future__ import annotations


def load_afm_dataset(file_path: str):
    """Load one IBW file using AFM-tools."""
    return _afm_viz().load_afm_dataset(file_path)


def preferred_channel_index(labels: list[str]) -> int:
    """Choose AFM-tools' preferred initial channel for a preview."""
    return int(_afm_viz().preferred_channel_index(labels))


def render_afm_preview(
    dataset,
    selected_channel_indices: list[int],
    *,
    show_metric_overlay: bool = False,
):
    """Render selected AFM/PFM channels with AFM-tools' default preview style."""
    afm_viz = _afm_viz()
    options = afm_viz.AfmPreviewOptions(
        selected_channel_indices=list(selected_channel_indices),
        show_metric_overlay=show_metric_overlay,
    )
    return afm_viz.render_afm_preview(dataset, options)


def _afm_viz():
    try:
        from afm_tools import afm_viz
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "AFM-tools with matplotlib is required to preview AFM/PFM files. "
            "Install the AFM-tools package into the same Python environment used to launch PLD Workflow."
        ) from exc

    required_names = ("AfmPreviewOptions", "load_afm_dataset", "preferred_channel_index", "render_afm_preview")
    missing = [name for name in required_names if not hasattr(afm_viz, name)]
    if missing:
        missing_names = ", ".join(f"afm_tools.afm_viz.{name}" for name in missing)
        raise RuntimeError(f"Upgrade AFM-tools. Missing required preview API: {missing_names}.")
    return afm_viz


__all__ = ["load_afm_dataset", "preferred_channel_index", "render_afm_preview"]

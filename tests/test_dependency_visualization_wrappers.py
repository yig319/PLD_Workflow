import sys
import types

import numpy as np
import pytest

from pld_workflow.raw import afm as afm_pfm_plotting
from pld_workflow.raw import xrd as xrd_rsm_visualization
from pld_workflow.raw import widgets as raw_widgets


def test_afm_plotting_wrapper_delegates_to_afm_tools(monkeypatch):
    fake_package = types.ModuleType("afm_tools")
    fake_viz = types.ModuleType("afm_tools.afm_viz")
    captured_options = {}

    class FakeOptions:
        def __init__(self, selected_channel_indices, show_metric_overlay=False):
            self.selected_channel_indices = selected_channel_indices
            self.show_metric_overlay = show_metric_overlay

    def _load_dataset(file_path):
        return types.SimpleNamespace(
            file_path=file_path,
            images=np.ones((2, 2, 1)),
            sample_name="sample",
            labels=["Height"],
            scan_size=None,
        )

    def _render_preview(dataset, options):
        captured_options["selected"] = options.selected_channel_indices
        captured_options["overlay"] = options.show_metric_overlay
        return types.SimpleNamespace(
            figure={"labels": dataset.labels},
            message=f"selected {options.selected_channel_indices}",
        )

    fake_viz.AfmPreviewOptions = FakeOptions
    fake_viz.load_afm_dataset = _load_dataset
    fake_viz.preferred_channel_index = lambda labels: labels.index("Height")
    fake_viz.render_afm_preview = _render_preview
    fake_package.afm_viz = fake_viz
    monkeypatch.setitem(sys.modules, "afm_tools", fake_package)
    monkeypatch.setitem(sys.modules, "afm_tools.afm_viz", fake_viz)

    dataset = afm_pfm_plotting.load_afm_dataset("sample.ibw")
    rendered = afm_pfm_plotting.render_afm_preview(
        dataset,
        [0],
        show_metric_overlay=True,
    )

    assert dataset.sample_name == "sample"
    assert afm_pfm_plotting.preferred_channel_index(dataset.labels) == 0
    assert captured_options == {
        "selected": [0],
        "overlay": True,
    }
    assert rendered.figure == {"labels": ["Height"]}
    assert rendered.message == "selected [0]"


def test_afm_plotting_wrapper_requires_upstream_renderer(monkeypatch):
    fake_package = types.ModuleType("afm_tools")
    fake_viz = types.ModuleType("afm_tools.afm_viz")
    fake_package.afm_viz = fake_viz
    monkeypatch.setitem(sys.modules, "afm_tools", fake_package)
    monkeypatch.setitem(sys.modules, "afm_tools.afm_viz", fake_viz)

    dataset = types.SimpleNamespace(
        file_path="sample.ibw",
        images=np.ones((4, 4, 1)),
        sample_name="sample",
        labels=["Height"],
        scan_size=None,
    )

    with pytest.raises(RuntimeError, match="Missing required preview API"):
        afm_pfm_plotting.render_afm_preview(dataset, [0])


def test_raw_visualization_accepts_afm_preview_result_object(monkeypatch):
    sentinel_figure = object()
    sentinel_pixmap = object()
    preview_result = types.SimpleNamespace(figure=sentinel_figure, message="AFM preview updated.")

    monkeypatch.setattr(raw_widgets, "_figure_to_pixmap", lambda obj: sentinel_pixmap if obj is sentinel_figure else None)

    assert raw_widgets._result_to_pixmap(preview_result) is sentinel_pixmap


def test_xrd_plotting_wrapper_delegates_to_xrd_utils(monkeypatch):
    fake_package = types.ModuleType("xrd_utils")
    fake_package.__path__ = []
    fake_viz = types.ModuleType("xrd_utils.xrd_viz")
    fake_viz.render_xrd_preview = lambda file_path: ("xrd_utils.xrd_viz.plot_xrd", {"file": file_path})
    fake_package.xrd_viz = fake_viz
    monkeypatch.setitem(sys.modules, "xrd_utils", fake_package)
    monkeypatch.setitem(sys.modules, "xrd_utils.xrd_viz", fake_viz)

    backend, figure = xrd_rsm_visualization.render_xrd_preview("scan.xrdml")

    assert backend == "xrd_utils.xrd_viz.plot_xrd"
    assert figure == {"file": "scan.xrdml"}

from pld_workflow.parameter_export import build_default_file_stem
from pld_workflow.raw_visualization import get_raw_file_spec


def test_build_default_file_stem_prefers_growth_id_and_user():
    assert build_default_file_stem("PLD-001", "Yichen", "04/02/2026") == "PLD-001_Yichen_04022026"


def test_raw_file_specs_cover_visualizer_inputs():
    assert ".ibw" in get_raw_file_spec("afm").extensions
    assert ".xrdml" in get_raw_file_spec("xrd").extensions
    assert ".xml" in get_raw_file_spec("rsm").extensions

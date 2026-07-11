# Examples

This folder is intentionally small. It only keeps the launcher scripts that map to the desktop apps we want to preserve for future Windows builds. The `launch_` prefix is intentional so these scripts are easy to distinguish from the package modules under `src/pld_workflow/apps/`:

- `launch_parameter_form.py`
- `launch_plume_manager.py`
- `launch_xrd_visualizer.py`
- `launch_afm_pfm_visualizer.py`
- `launch_rheed_imm_visualizer.py`

Run them from the repository root:

```bash
python examples/launch_parameter_form.py
python examples/launch_plume_manager.py
python examples/launch_xrd_visualizer.py
python examples/launch_afm_pfm_visualizer.py
python examples/launch_rheed_imm_visualizer.py
```

Notebook-based demos and analysis examples now live under the top-level `notebooks/` folder instead of `examples/`.

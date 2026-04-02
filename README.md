# pldflow

`pldflow` is a desktop workflow package for pulsed laser deposition work. The cleaned repo now centers on three day-to-day desktop apps:

- `pld-parameter-form` for recording PLD growth parameters
- `pld-plume-manager` for packing plume image folders and attaching/editing metadata
- `pld-raw-visualizer` for quick XRD, RSM, and AFM raw-data previews

Optional analysis utilities live separately so the common recorder workflow stays lightweight.

## Install from a Git clone

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[analysis,visualization,build,docs]"
```

If you only want to fix the visualizer backend on an existing environment, this is the minimum command:

```bash
python -m pip install AFM-tools XRD-utils xrayutilities matplotlib
```

## Launch the apps

Console entry points:

```bash
pld-parameter-form
pld-plume-manager
pld-raw-visualizer
```

Source-tree launchers:

```bash
python examples/pld_app_parameter.py
python examples/pld_app_plume.py
python examples/pld_app_visualizer.py
```

Optional analysis runner:

```bash
python scripts/run_parameter_trend_demo.py
```

## Notes on scope

- The parameter recorder saves JSON and HTML reports.
- The plume manager is intentionally separate from the recorder, but it can load and edit recorder JSON metadata before packing or uploading.
- The parameter trend analyzer is also separate from the recorder and is meant for ad hoc review of historical JSON records.
- The raw visualizer supports XRD scan, RSM, and AFM file selection in the UI, but the actual plotting backend still depends on whichever XRD/AFM helper packages are installed in your environment.

## Windows executable builds

PyInstaller support is driven by:

```powershell
./scripts/build_windows_exe.ps1 -App parameter
./scripts/build_windows_exe.ps1 -App plume
./scripts/build_windows_exe.ps1 -App visualizer
```

Add `-OneFile` if you want a single-file executable.

## Repository layout

- `src/pld_workflow/` contains the maintained source code
- `examples/` contains the three launcher scripts kept for future `.exe` packaging
- `notebooks/` contains organized analysis and workflow demos
- `scripts/` contains helper launch/build scripts that are not part of the default app surface

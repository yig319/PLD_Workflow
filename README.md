# pldflow

`pldflow` is a desktop workflow package for pulsed laser deposition work. The cleaned repo now centers on the day-to-day desktop apps:

- `pld-parameter-form` for recording PLD growth parameters
- `pld-plume-manager` for packing plume image folders and attaching/editing metadata
- `pld-xrd-visualizer` for quick XRD-family raw-data previews, including RSM
- `pld-afm-pfm-visualizer` for AFM/PFM channel review and roughness-aware previews
- `pld-rheed-imm-visualizer` for lightweight IMM movie inspection

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
pld-xrd-visualizer
pld-afm-pfm-visualizer
pld-rheed-imm-visualizer
pld-raw-visualizer
pld-afm-visualizer
```

Source-tree launchers:

```bash
python examples/parameter_form.py
python examples/plume_manager.py
python examples/xrd_visualizer.py
python examples/afm_pfm_visualizer.py
python examples/rheed_imm_visualizer.py
```

Optional analysis runner:

```bash
python scripts/run_parameter_trend_demo.py
```

## Notes on scope

- The parameter recorder saves JSON and HTML reports.
- The plume manager is intentionally separate from the recorder, but it can load and edit recorder JSON metadata before packing or uploading.
- The parameter trend analyzer is also separate from the recorder and is meant for ad hoc review of historical JSON records.
- The XRD visualizer supports XRD scan and RSM file selection in the UI.
- The AFM/PFM visualizer supports quick single-channel review plus a comprehensive multi-channel mode.
- The actual plotting backend still depends on whichever XRD/AFM helper packages are installed in your environment.

## Windows executable builds

The recommended flow for Windows builds is:

1. Clean old build artifacts.
2. Build with the Python helper.

Clean old artifacts first:

```bash
python scripts/clean_build_artifacts.py
```

This removes old `build/` and `dist/` outputs when they are not locked. If cleanup fails, close any running PLD `.exe`, close Explorer windows opened inside `build/` or `dist/`, and pause Dropbox sync before retrying.

Build one-file executables with the Python helper:

```bash
python scripts/build_windows_exe.py --app parameter_form --onefile
python scripts/build_windows_exe.py --app plume_manager --onefile
python scripts/build_windows_exe.py --app xrd_visualizer --onefile
python scripts/build_windows_exe.py --app afm_pfm_visualizer --onefile
python scripts/build_windows_exe.py --app rheed_imm_visualizer --onefile
```

If you want to force a specific interpreter:

```bash
C:\Users\yichen\anaconda3\envs\pld\python.exe scripts\build_windows_exe.py --app xrd_visualizer --onefile --python C:\Users\yichen\anaconda3\envs\pld\python.exe
```

Expected outputs:

- `dist/PLDParameterForm.exe`
- `dist/PLDPlumeManager.exe`
- `dist/PLDXRDVisualizer.exe`
- `dist/PLDAFMPFMVisualizer.exe`
- `dist/PLDRHEEDIMMVisualizer.exe`

Notes:

- The Python build helper now performs the PyInstaller work in a temporary folder outside the repo and only copies the final fresh result back into `dist/`. This avoids stale `build/` folders and reduces Dropbox file-lock issues.
- The build helper still accepts legacy aliases like `parameter`, `plume`, `visualizer`, and `afm_visualizer`, but the newer direct names are the recommended ones.
- The diffraction visualizer build explicitly bundles `XRD-utils` and `xrayutilities`.
- The AFM visualizer build explicitly bundles `AFM-tools`.

## Repository layout

- `src/pld_workflow/` contains the maintained source code
- `examples/` contains the launcher scripts kept for future `.exe` packaging
- `notebooks/` contains organized analysis and workflow demos
- `scripts/` contains helper launch/build scripts that are not part of the default app surface

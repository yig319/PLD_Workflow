# pldflow

`pldflow` is a desktop workflow package for pulsed laser deposition work. The cleaned repo now centers on the day-to-day desktop apps:

- `pld-parameter-form` for recording PLD growth parameters
- `pld-plume-manager` for creating plume workspaces from PLD JSON metadata, staging raw files, previewing frames, and packing/editing metadata
- `pld-xrd-visualizer` for quick XRD-family raw-data previews, including RSM
- `pld-afm-pfm-visualizer` for AFM/PFM channel review and roughness-aware previews
- `pld-rheed-imm-visualizer` for lightweight IMM movie inspection

Optional analysis utilities live separately so the common recorder workflow stays lightweight.

## Install from a Git clone

### With Conda (recommended on Linux)

```bash
conda env create -f environment.yml
conda activate pld
python -m pip install -e ".[analysis,visualization,build,docs]"
```

### With pip only

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[analysis,visualization,build,docs]"
```

> **Linux note:** On Linux, installing PyQt5 via pip may fail with an xcb platform plugin error. Install PyQt5 from conda instead (`conda install -c conda-forge pyqt=5`), or install the missing system library: `sudo apt install libxcb-cursor0`.

If you only want to fix the visualizer backend on an existing environment, this is the minimum command:

```bash
python -m pip install "AFM-tools>=2.1.0" XRD-utils xrayutilities matplotlib
```

## WSL display setup

If running on WSL, Qt apps need a display path. Choose one option:

### WSLg (recommended — Win 11 / recent Win 10)

WSLg is built in. Add this line to `~/.bashrc` so it applies every session:

```bash
echo 'export QT_QPA_PLATFORM=wayland' >> ~/.bashrc
```

Then reload with `source ~/.bashrc` or open a new terminal.

### X server fallback

If WSLg isn't available, install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) on Windows, launch it, then add to `~/.bashrc`:

```bash
echo 'export DISPLAY=$(ip route | awk '\''/default/ {print $3}'\''):0' >> ~/.bashrc
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
python examples/launch_parameter_form.py
python examples/launch_plume_manager.py
python examples/launch_xrd_visualizer.py
python examples/launch_afm_pfm_visualizer.py
python examples/launch_rheed_imm_visualizer.py
```

## App details

### Parameter Form
Records PLD growth parameters. Saves JSON and exports HTML reports. The form can open an
existing record for editing or use it as a template for a new run. Template mode preserves
reusable process settings while clearing run-specific pulse values and starting a fresh
timestamp. Per-target pulse tracking supports:

- Target ID plus material and chamber matching
- Local JSON-history lookup using records earlier than the form timestamp
- Separate pre-ablation, growth-ablation, additional on-target, and off-target pulse counts
- Calculated before-run, after-run, on-target, and all-laser totals
- Verified manual correction with a recorded reason

### Plume Manager
Handles plume workspace creation from PLD JSON metadata — staging raw files, previewing frames, and packing/editing metadata. Features:
- Create target folders from a PLD JSON record, including `<target>_Pre` folders when pre-ablation pulses are listed
- Workspace browser for `target/raw` and `target/BMP/plume/frame` folders
- H5 browser to inspect packed archive structure and preview individual frames
- Load and edit recorder JSON metadata before packing

### XRD Visualizer
Quick XRD/RSM raw-data previews. Supports XRD scan and RSM file selection in the UI. Backend depends on XRD helper packages (`XRD-utils`, `xrayutilities`).

### AFM/PFM Visualizer
AFM/PFM channel review with roughness-aware previews. Supports quick single-channel review plus a comprehensive multi-channel mode. Backend depends on AFM helper packages (`AFM-tools`).

### RHEED/IMM Visualizer
Lightweight IMM movie inspection.

### Parameter Trend Analyzer
Ad hoc review of historical JSON records. Runs separately from the recorder:

```bash
python scripts/run_parameter_trend_demo.py
```

## Windows executable builds

The recommended flow for Windows builds is:

1. Clean old build artifacts.
2. Build with the Python helper from a small app-specific environment.

The detailed packaging guide is in [docs/source/building.rst](docs/source/building.rst). It includes:

- the minimal package list for each app
- recommended `venv` commands for smaller executables
- the equivalent Conda workflow if you prefer Conda
- per-app build commands using `--python`

Tested plume-manager `venv` build, without activating the environment:

```powershell
python -m venv .venv-build-plume
.\.venv-build-plume\Scripts\python.exe -m pip install -r requirements-build\plume_manager.txt
.\.venv-build-plume\Scripts\python.exe scripts\build_windows_exe.py --app plume_manager --onefile --python .\.venv-build-plume\Scripts\python.exe
```

That command flow writes the executable to:

```text
dist/PLDPlumeManager.exe
```

If `pip` inside a fresh `venv` is broken or partially upgraded because of a Windows/Dropbox file lock, repair it with:

```powershell
.\.venv-build-plume\Scripts\python.exe -m ensurepip --upgrade
```

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

If your goal is a smaller executable, prefer a dedicated build environment per app instead of a single large all-purpose environment. The package sets used for packaging are also listed in:

- `requirements-build/parameter_form.txt`
- `requirements-build/plume_manager.txt`
- `requirements-build/xrd_visualizer.txt`
- `requirements-build/afm_pfm_visualizer.txt`
- `requirements-build/rheed_imm_visualizer.txt`

The same no-activation pattern works for the other apps too. Replace the environment name, requirements file, and `--app` value. For example:

```powershell
python -m venv .venv-build-parameter
.\.venv-build-parameter\Scripts\python.exe -m pip install -r requirements-build\parameter_form.txt
.\.venv-build-parameter\Scripts\python.exe scripts\build_windows_exe.py --app parameter_form --onefile --python .\.venv-build-parameter\Scripts\python.exe
```

Expected outputs:

- `dist/PLDParameterForm.exe`
- `dist/PLDPlumeManager.exe`
- `dist/PLDXRDVisualizer.exe`
- `dist/PLDAFMPFMVisualizer.exe`
- `dist/PLDRHEEDIMMVisualizer.exe`

Build notes:

- The build helper performs PyInstaller work in a temporary folder outside the repo and only copies the final result into `dist/`, avoiding stale `build/` folders and Dropbox file-lock issues.
- Legacy aliases (`parameter`, `plume`, `visualizer`, `afm_visualizer`) are still accepted, but the new direct names are recommended.
- The diffraction visualizer build explicitly bundles `XRD-utils` and `xrayutilities`.
- The AFM visualizer build explicitly bundles `AFM-tools`.

## Sharing guidance

- For collaborators who already use Python, the most robust option is usually to share the repo plus a short install/build command list instead of a prebuilt `.exe`.
- For Windows users who do not want to install Python, build a `.exe` and share it outside Git history, for example as a GitHub Release asset, Dropbox/OneDrive file, or a zipped `dist/` artifact.
- Large single-file PyInstaller outputs often exceed practical Git repository size limits, so avoiding committed binaries is the right default.

## Repository layout

- `src/pld_workflow/apps/` contains the Qt application entry points.
- `src/pld_workflow/raw/` contains raw-data preview adapters and drag/drop widgets.
- `src/pld_workflow/` keeps shared domain code, archive helpers, analysis helpers, and reusable Qt form logic.
- `examples/` contains the launcher scripts kept for future `.exe` packaging
- `notebooks/` contains organized analysis and workflow demos
- `scripts/` contains helper launch/build scripts that are not part of the default app surface

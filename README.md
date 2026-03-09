# PLD_Workflow

Desktop application for recording PLD growth parameters.

## Quick Start (local Git checkout)

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
pld-parameter-form
```

Alternative launch commands:

```bash
python -m pld_workflow
python examples/pld_app_parameter.py
```

## Fresh Conda Env (pip-preferred)

Use this when starting from a clean machine/user environment.

```bash
conda create -n pld_clean python=3.10 -y
conda activate pld_clean

# run from repository root
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-pip.txt
```

Run apps:

```bash
python examples/pld_app_parameter.py
python examples/pld_app_visualizer.py
```

Important:

- Keep this environment pip-managed for Qt/PyQt (do not mix `conda install pyqt` with `pip install PyQt5` in the same env).
- In VS Code, select the interpreter from this env (for example: `/home/yichen/anaconda3/envs/pld_clean/bin/python`).
- `requirements-pip.txt` installs runtime + visualizer + build dependencies.

## Raw XRD/AFM Visualization

Raw-data visualization is now a **separate app** from the parameter form.
Launch it with:

```bash
pld-raw-visualizer
```

or from source:

```bash
python examples/pld_app_visualizer.py
```

The visualizer provides three square blocks (XRD scan, RSM, and AFM) with:

- drag-and-drop loading
- clickable drag area to open file dialog
- embedded preview (when the visualizer returns an image/figure/array)
- `Copy Image` button for clipboard paste into OneNote/PowerPoint/docs
- `Export PNG` button to save the current preview image

File expectations:

- AFM: `.ibw`
- XRD scan: `.xrdml` (and compatible xrdutilities-readable formats)
- RSM: `.xrdml` / `.xml`

Note: `XRD-utils` uses `xrayutilities` for file loading. If your environment
does not include it, install `xrayutilities` as well.

If the external package opens its own plot window and does not return image data,
the app shows a status message and keeps using the external window behavior.

All-in-one install (runtime + visualizer + build):

```bash
python -m pip install -r requirements-pip.txt
```

Visualization-only install:

```bash
python -m pip install -e ".[visualization]"
python -m pip install xrayutilities
```

### Linux Troubleshooting (`GLIBCXX_3.4.29 not found`)

If AFM/XRD visualization fails with a message like `GLIBCXX_3.4.29 not found`,
your C++ runtime does not match the wheel binaries.

Use one conda environment consistently and refresh runtime libs:

```bash
conda activate pld
conda install -n pld -c conda-forge libstdcxx-ng libgcc-ng
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  numpy scipy matplotlib AFM-tools XRD-utils xrayutilities
```

If you launch from VS Code, select interpreter:

- `/home/yichen/anaconda3/envs/pld/bin/python`

### `mayavi` install failure (`Failed building wheel for mayavi`)

This is common on Linux when installing with `pip` because it tries to compile
VTK/Mayavi locally. For this app, `mayavi` is not required for the AFM 2D path.

If you still need `mayavi` for 3D workflows, install via conda binaries:

```bash
conda activate pld
conda install -n pld -c conda-forge mayavi vtk pyqt
```

## Windows `.exe` Build

For users who should run without installing Python, build and distribute an executable:

```powershell
./scripts/build_windows_exe.ps1
```

Single-file build:

```powershell
./scripts/build_windows_exe.ps1 -OneFile
```

Manual build command (directly from the Python entry file):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
python -m PyInstaller --noconfirm --clean --windowed --name PLDParameterForm --paths src src/pld_workflow/app.py
```

Output:

- Folder build: `dist/PLDParameterForm/PLDParameterForm.exe`
- Single file (if adding `--onefile`): `dist/PLDParameterForm.exe`

Platform note:

- PyInstaller builds for the current OS.
- Running the build on Linux does **not** create a Windows `.exe`.
- To get a Windows `.exe`, run the build on Windows (or a Windows CI runner).

Detailed instructions are in [DISTRIBUTION.md](DISTRIBUTION.md).

## Linux Build (for local testing)

On Linux, you can build a Linux executable:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
python -m PyInstaller --noconfirm --clean --windowed --name PLDParameterForm --paths src src/pld_workflow/app.py
```

Linux output:

- `dist/PLDParameterForm/PLDParameterForm`

## Current App Scope

- Parameter recording form only.
- JSON and HTML export through "Save Parameters".
- Output file is saved under the path in the "Directory" field.

## Camera Position Calibration (lab workflow)

1. Open software `HPV-X` on desktop.
2. Click `Live` and increase `EXPOSE` to `10,000,000 ns` to align focus.
3. Decrease `EXPOSE` to `2,000,000 ns` and click `REC` before ablation.

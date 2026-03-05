# PLD_Workflow

Desktop application for recording PLD growth parameters.

## Quick Start (local Git checkout)

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install .
pld-parameter-form
```

Alternative launch commands:

```bash
python -m pld_workflow
python examples/pld_app_parameter.py
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
python -m pip install ".[build]"
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
python -m pip install ".[build]"
python -m PyInstaller --noconfirm --clean --windowed --name PLDParameterForm --paths src src/pld_workflow/app.py
```

Linux output:

- `dist/PLDParameterForm/PLDParameterForm`

## Current App Scope

- Parameter recording form only.
- JSON export through "Save Parameters".
- Output file is saved under the path in the "Directory" field.

## Camera Position Calibration (lab workflow)

1. Open software `HPV-X` on desktop.
2. Click `Live` and increase `EXPOSE` to `10,000,000 ns` to align focus.
3. Decrease `EXPOSE` to `2,000,000 ns` and click `REC` before ablation.

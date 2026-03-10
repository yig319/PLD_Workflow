# Distribution and Local Installation

This repository now supports two workflows:

1. Install and run locally from a downloaded Git repository.
2. Build a Windows `.exe` so end users do not need Python installed.

## 1. Local install from Git checkout

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
```

Run the app:

```bash
pld-parameter-form
```

Alternative:

```bash
python -m pld_workflow
```

## 2. Windows executable build (`.exe`)

Important: Python is needed on the build machine only. End users running the final `.exe` do not need Python.

### Option A: Python build script (recommended)

From the repository root:

```powershell
python scripts/build_windows_exe.py
```

For a single-file executable:

```powershell
python scripts/build_windows_exe.py --onefile
```

Build output:

- Folder mode:
  - `dist/PLDParameterForm/PLDParameterForm.exe`
  - `dist/PLDRawVisualizer/PLDRawVisualizer.exe`
- One-file mode:
  - `dist/PLDParameterForm.exe`
  - `dist/PLDRawVisualizer.exe`

Pre-build smoke test using the same PyInstaller entry wrappers:

```powershell
python scripts/pyinstaller_entry_pld_form.py
python scripts/pyinstaller_entry_pld_visualizer.py
```

### Option B: Manual PyInstaller command

```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall ".[visualization,build]" xrayutilities
python -m PyInstaller --noconfirm --clean --windowed --collect-all PyQt5 --distpath dist --workpath build --specpath build/spec --paths src --name PLDParameterForm scripts/pyinstaller_entry_pld_form.py
python -m PyInstaller --noconfirm --clean --windowed --collect-all PyQt5 --distpath dist --workpath build --specpath build/spec --paths src --name PLDRawVisualizer scripts/pyinstaller_entry_pld_visualizer.py
```

Run executables from `dist/`, not from `build/` (temporary files).

## 3. Publishing suggestion

For Windows users, publish the built artifacts from `dist/` as a GitHub Release asset.
Add code signing later to reduce SmartScreen warnings.

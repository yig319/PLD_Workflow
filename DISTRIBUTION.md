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

### Option A: One command script (recommended)

From PowerShell in the repository root:

```powershell
./scripts/build_windows_exe.ps1
```

For a single-file executable:

```powershell
./scripts/build_windows_exe.ps1 -OneFile
```

Build output:

- Folder mode: `dist/PLDParameterForm/PLDParameterForm.exe`
- One-file mode: `dist/PLDParameterForm.exe`

### Option B: Manual PyInstaller command

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-pip.txt
python -m PyInstaller --noconfirm --clean --windowed --name PLDParameterForm --collect-all PyQt5 --paths src src/pld_workflow/app.py
```

## 3. Publishing suggestion

For Windows users, publish the built artifacts from `dist/` as a GitHub Release asset.
Add code signing later to reduce SmartScreen warnings.

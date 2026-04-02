# Distribution and Local Installation

This repository supports two common workflows:

1. Run the apps locally from a Git checkout.
2. Build Windows `.exe` bundles for end users.

## 1. Local install from Git checkout

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,visualization,build,docs]"
```

Console entry points:

```bash
pld-parameter-form
pld-plume-manager
pld-raw-visualizer
```

## 2. Windows executable builds

From PowerShell in the repository root:

```powershell
./scripts/build_windows_exe.ps1 -App parameter
./scripts/build_windows_exe.ps1 -App plume
./scripts/build_windows_exe.ps1 -App visualizer
```

For a single-file executable:

```powershell
./scripts/build_windows_exe.ps1 -App parameter -OneFile
```

Output names:

- `parameter` -> `dist/PLDParameterForm/` or `dist/PLDParameterForm.exe`
- `plume` -> `dist/PLDPlumeManager/` or `dist/PLDPlumeManager.exe`
- `visualizer` -> `dist/PLDRawVisualizer/` or `dist/PLDRawVisualizer.exe`

## 3. Publishing suggestion

For Windows users, publish the built artifacts from `dist/` as GitHub Release assets.
Add code signing later if SmartScreen warnings become a problem.

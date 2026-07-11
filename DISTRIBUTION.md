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
pld-xrd-visualizer
pld-afm-pfm-visualizer
pld-rheed-imm-visualizer
```

## 2. Windows executable builds

From PowerShell in the repository root:

```powershell
./scripts/build_windows_exe.ps1 -App parameter_form
./scripts/build_windows_exe.ps1 -App plume_manager
./scripts/build_windows_exe.ps1 -App xrd_visualizer
./scripts/build_windows_exe.ps1 -App afm_pfm_visualizer
./scripts/build_windows_exe.ps1 -App rheed_imm_visualizer
```

For a single-file executable:

```powershell
./scripts/build_windows_exe.ps1 -App plume_manager -OneFile
```

Output names:

- `parameter_form` -> `dist/PLDParameterForm/` or `dist/PLDParameterForm.exe`
- `plume_manager` -> `dist/PLDPlumeManager/` or `dist/PLDPlumeManager.exe`
- `xrd_visualizer` -> `dist/PLDXRDVisualizer/` or `dist/PLDXRDVisualizer.exe`
- `afm_pfm_visualizer` -> `dist/PLDAFMPFMVisualizer/` or `dist/PLDAFMPFMVisualizer.exe`
- `rheed_imm_visualizer` -> `dist/PLDRHEEDIMMVisualizer/` or `dist/PLDRHEEDIMMVisualizer.exe`

## 3. Publishing suggestion

- Prefer sharing the source repo plus install/build instructions when collaborators already have Python available.
- For non-Python end users, publish the built artifacts from `dist/` as GitHub Release assets or upload the zipped output folder to shared storage such as Dropbox or OneDrive.
- Avoid committing large `.exe` files into Git history.
- Add code signing later if SmartScreen warnings become a problem.

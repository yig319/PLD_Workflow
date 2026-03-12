# Distribution and Release Notes

## Local development install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Build the package locally

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Automated GitHub release flow

The repository workflow is `.github/workflows/main.yml`.

- every push and pull request to `main` runs a build check
- pushes to `main` with `#major`, `#minor`, or `#patch` also create a version tag and publish to PyPI
- tags use semantic versioning: `vX.Y.Z`

Release markers:

- `#major` -> `+1.0.0`
- `#minor` -> `+0.1.0`
- `#patch` -> `+0.0.1`

If the same commit is retried, the workflow reuses an existing tag on `HEAD` instead of bumping again.

## PyPI setup required once

For automated publishing to work, configure a trusted publisher on PyPI for project `pldflow` and point it to this GitHub repository and workflow.

## Windows executable build

Use the helper script from the repository root:

```powershell
python scripts/build_windows_exe.py
```

Single-file build:

```powershell
python scripts/build_windows_exe.py --onefile
```

Manual PyInstaller path:

```powershell
python -m pip install -e ".[build,visualization]"
python -m PyInstaller --noconfirm --clean --windowed --collect-all PyQt5 --distpath dist --workpath build --specpath build/spec --paths src --name PLDParameterForm scripts/pyinstaller_entry_pld_form.py
python -m PyInstaller --noconfirm --clean --windowed --collect-all PyQt5 --distpath dist --workpath build --specpath build/spec --paths src --name PLDRawVisualizer scripts/pyinstaller_entry_pld_visualizer.py
```

Build outputs are written under `dist/`.

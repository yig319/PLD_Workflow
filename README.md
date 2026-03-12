# pldflow

`pldflow` is a desktop workflow package for PLD parameter capture and raw-data visualization.

PyPI package name: `pldflow`  
Python import package: `pld_workflow`

## Install from PyPI

```bash
python -m pip install pldflow
```

This installs the parameter form and exposes:

```bash
pld-parameter-form
```

## Development install from a Git clone

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

This gives you an editable install of the package plus the optional analysis, visualization, docs, and build dependencies.

Launch commands:

```bash
pld-parameter-form
pld-raw-visualizer
python -m pld_workflow
python examples/pld_app_parameter.py
python examples/pld_app_visualizer.py
```

## How `from pld_workflow.app import main` works

That import does not by itself mean the package is editable-installed. It works in three cases:

1. The package is installed normally with `pip install .` or `pip install pldflow`.
2. The package is installed in editable mode with `pip install -e .`.
3. The caller adds `src/` to `sys.path` before importing.

In this repository, `examples/pld_app_parameter.py` uses option 3. It prepends `src/` to `sys.path`, so you can run the example directly from the repo root even if the package is not installed. Using `pip install -r requirements-dev.txt` is still the cleaner setup for daily work.

## Visualization dependencies

The raw-data visualizer depends on the companion packages:

- `AFM-tools`
- `XRD-utils`
- `xrayutilities`

Those are included by `requirements-dev.txt` and by the package extra:

```bash
python -m pip install -e ".[visualization]"
```

If you only want the form application, `PyQt5` is the only required runtime dependency.

## Windows executable build

PyInstaller build support is included through the `build` extra:

```bash
python -m pip install -e ".[build,visualization]"
python scripts/build_windows_exe.py
```

Detailed packaging notes are in [DISTRIBUTION.md](DISTRIBUTION.md).

## Release workflow

GitHub Actions now follows the same release markers used in the AFM and XRD repos.

- `#major` bumps `+1.0.0`
- `#minor` bumps `+0.1.0`
- `#patch` bumps `+0.0.1`
- no marker runs CI only and does not publish

The workflow builds on every push and pull request to `main`. When a push to `main` contains one of the release markers, it creates the next `vX.Y.Z` tag and publishes to PyPI.

Before the first automated publish, configure PyPI trusted publishing for the `pldflow` project so GitHub Actions is allowed to upload releases.

## Read the Docs

Docs build from `docs/source/` using `.readthedocs.yaml`. Read the Docs should install:

- `docs/requirements.txt`
- the package itself from the repo root

## Current scope

- Parameter form for PLD growth records
- JSON and HTML export
- Standalone raw-data visualizer for XRD, RSM, and AFM inputs

# Examples

Use the top-level README for the primary install instructions.

From a fresh clone, the recommended setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## `pld_app_parameter.py`

Runs the PLD parameter form.

```bash
python examples/pld_app_parameter.py
```

Installed command:

```bash
pld-parameter-form
```

## `pld_app_visualizer.py`

Runs the raw-data visualizer for AFM, XRD scan, and RSM inputs.

```bash
python examples/pld_app_visualizer.py
```

Installed command:

```bash
pld-raw-visualizer
```

`pld_app_parameter.py` prepends `src/` to `sys.path`, so it can import `pld_workflow.app` directly from the source tree without requiring an installed package.

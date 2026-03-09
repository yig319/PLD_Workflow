# Examples

## `pld_app_parameter.py`

Compatibility launcher for the parameter form.

```bash
python examples/pld_app_parameter.py
```

Preferred installed command:

```bash
pld-parameter-form
```

## `pld_app_visualizer.py`

Standalone raw-data visualizer for:

- AFM `.ibw`
- XRD scan files
- RSM `.xrdml` / `.xml`

```bash
python examples/pld_app_visualizer.py
```

Preferred installed command:

```bash
pld-raw-visualizer
```

Inside each block, use:

- `Copy Image` to copy the embedded figure to clipboard
- `Export PNG` to save a figure file

Optional raw-data visualization dependencies:

```bash
python -m pip install ".[visualization]"
```

If Linux shows `GLIBCXX_3.4.29 not found`, run:

```bash
conda activate pld
conda install -n pld -c conda-forge libstdcxx-ng libgcc-ng
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  numpy scipy matplotlib AFM-tools XRD-utils xrayutilities
```

If `pip install mayavi` fails with `Failed building wheel for mayavi`:

- `mayavi` is optional for this visualizer app.
- For 3D tooling, install with conda instead:

```bash
conda activate pld
conda install -n pld -c conda-forge mayavi vtk pyqt
```

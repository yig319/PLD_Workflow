# Tests

This folder keeps small regression tests for the maintained PLD workflow surface:

- `test_smoke.py` checks very small package basics, including default export naming and supported raw-data file extensions.
- `test_analysis.py` checks JSON-record discovery and parameter trend table construction.
- `test_plume_management.py` checks plume image-folder packing into the HDF5 archive format.
- `test_dependency_visualization_wrappers.py` checks that PLD delegates AFM/XRD plotting to AFM-tools and XRD-utils when possible, while still supporting older installed AFM-tools versions.

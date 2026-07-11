Build Guide
===========

This project can build each desktop app into its own Windows executable with ``PyInstaller``.

The most important packaging rule is simple:

- Build each app from a small environment that contains only the packages that app actually needs.

If you build from a large all-purpose Conda environment, ``PyInstaller`` often discovers extra notebook, docs, plotting, or scientific packages and the final ``.exe`` becomes much larger than necessary.

Recommended strategy
--------------------

For the smallest and cleanest Windows builds:

1. Use a separate build environment for each app, or at least separate light apps from heavy apps.
2. Prefer a small ``venv`` over a large shared Conda environment when possible.
3. Keep your main development environment separate from your packaging environments.

Why separate environments help
------------------------------

- ``parameter_form`` only needs ``PyQt5`` and ``PyInstaller``.
- ``plume_manager`` needs ``PyQt5``, ``h5py``, ``numpy``, ``Pillow``, and ``PyInstaller``.
- ``xrd_visualizer`` needs diffraction packages that the lighter apps do not need.
- ``afm_pfm_visualizer`` needs AFM tooling that the lighter apps do not need.
- ``rheed_imm_visualizer`` is also lighter than the XRD and AFM builds.

Building every app from one large environment works, but it usually makes the executables bigger.

Minimal package lists
---------------------

The repository includes one packaging requirements file per app:

- ``requirements-build/parameter_form.txt``
- ``requirements-build/plume_manager.txt``
- ``requirements-build/xrd_visualizer.txt``
- ``requirements-build/afm_pfm_visualizer.txt``
- ``requirements-build/rheed_imm_visualizer.txt``

These files list the third-party packages needed to build each executable. The local project code itself is loaded directly from ``src/`` by ``scripts/build_windows_exe.py``.

Recommended Windows ``venv`` workflow
-------------------------------------

Example: build the plume manager from a small dedicated environment.

You do not need to activate the environment if you prefer to call its interpreter directly.

.. code-block:: powershell

   python -m venv .venv-build-plume
   .\.venv-build-plume\Scripts\python.exe -m pip install -r requirements-build\plume_manager.txt
   .\.venv-build-plume\Scripts\python.exe scripts\build_windows_exe.py --app plume_manager --onefile --python .\.venv-build-plume\Scripts\python.exe

This writes the final executable to:

.. code-block:: text

   dist/PLDPlumeManager.exe

You can repeat the same pattern for other apps by changing the environment name, the requirements file, and the ``--app`` value.

If ``pip`` inside a fresh ``venv`` is broken or partially upgraded because of a Windows/Dropbox file lock, repair it with:

.. code-block:: powershell

   .\.venv-build-plume\Scripts\python.exe -m ensurepip --upgrade

Example commands for each app
-----------------------------

Parameter form:

.. code-block:: powershell

   python -m venv .venv-build-parameter
   .\.venv-build-parameter\Scripts\python.exe -m pip install -r requirements-build\parameter_form.txt
   .\.venv-build-parameter\Scripts\python.exe scripts\build_windows_exe.py --app parameter_form --onefile --python .\.venv-build-parameter\Scripts\python.exe

Plume manager:

.. code-block:: powershell

   python -m venv .venv-build-plume
   .\.venv-build-plume\Scripts\python.exe -m pip install -r requirements-build\plume_manager.txt
   .\.venv-build-plume\Scripts\python.exe scripts\build_windows_exe.py --app plume_manager --onefile --python .\.venv-build-plume\Scripts\python.exe

XRD visualizer:

.. code-block:: powershell

   python -m venv .venv-build-xrd
   .\.venv-build-xrd\Scripts\python.exe -m pip install -r requirements-build\xrd_visualizer.txt
   .\.venv-build-xrd\Scripts\python.exe scripts\build_windows_exe.py --app xrd_visualizer --onefile --python .\.venv-build-xrd\Scripts\python.exe

AFM/PFM visualizer:

.. code-block:: powershell

   python -m venv .venv-build-afm
   .\.venv-build-afm\Scripts\python.exe -m pip install -r requirements-build\afm_pfm_visualizer.txt
   .\.venv-build-afm\Scripts\python.exe scripts\build_windows_exe.py --app afm_pfm_visualizer --onefile --python .\.venv-build-afm\Scripts\python.exe

RHEED IMM visualizer:

.. code-block:: powershell

   python -m venv .venv-build-rheed
   .\.venv-build-rheed\Scripts\python.exe -m pip install -r requirements-build\rheed_imm_visualizer.txt
   .\.venv-build-rheed\Scripts\python.exe scripts\build_windows_exe.py --app rheed_imm_visualizer --onefile --python .\.venv-build-rheed\Scripts\python.exe

Using Conda instead
-------------------

If you prefer Conda, the same idea still applies:

- create a separate Conda environment for each app
- install only the packages from that app's ``requirements-build/*.txt`` file
- run the build script with that environment's Python via ``--python``

Example:

.. code-block:: powershell

   conda create -n pld-plume-build python=3.13 -y
   conda activate pld-plume-build
   python -m pip install --upgrade pip
   python -m pip install -r requirements-build\plume_manager.txt
   python scripts\build_windows_exe.py --app plume_manager --onefile --python python

Conda is perfectly usable, but a large existing scientific Conda environment often produces larger executables than a clean ``venv``.

How the build helper works
--------------------------

``scripts/build_windows_exe.py`` does three main things:

1. installs the declared third-party dependencies for the selected app
2. runs ``PyInstaller`` in a temporary folder outside the repo
3. copies only the fresh final result back into ``dist/``

This helps reduce stale build artifacts and avoids many file-lock problems in Dropbox-synced folders.

Outputs
-------

Expected one-file outputs:

- ``dist/PLDParameterForm.exe``
- ``dist/PLDPlumeManager.exe``
- ``dist/PLDXRDVisualizer.exe``
- ``dist/PLDAFMPFMVisualizer.exe``
- ``dist/PLDRHEEDIMMVisualizer.exe``

Troubleshooting
---------------

If cleanup or copy-back fails:

- close any running PLD executable
- close Explorer windows opened inside ``dist/`` or ``build/``
- pause Dropbox sync temporarily
- rerun the build command

If the executable is still too large:

1. rebuild from a fresh dedicated ``venv``
2. avoid building from a large notebook-heavy Conda environment
3. confirm you are only installing the matching ``requirements-build/*.txt`` file for that app

If you want to inspect or reset old packaging outputs first:

.. code-block:: powershell

   python scripts\clean_build_artifacts.py

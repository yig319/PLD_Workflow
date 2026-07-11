Usage
=====

Install from PyPI
-----------------

.. code-block:: bash

   python -m pip install pldflow

Development install from a clone
--------------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e ".[analysis,visualization,build,docs]"

Run the applications
--------------------

.. code-block:: bash

   pld-parameter-form
   pld-plume-manager
   pld-raw-visualizer
   python -m pld_workflow

About source-tree imports
-------------------------

Use the app-specific package modules for direct imports, for example ``from pld_workflow.apps.parameter_form import main`` or ``from pld_workflow.apps.xrd_visualizer import main``.

The example launchers in ``examples/`` add ``src/`` to ``sys.path`` first, so they can run from a repository checkout without an installation step.

The plume manager workflow is designed around a workspace root that contains per-target folders with ``raw/`` and ``BMP/`` subfolders, plus an optional packed H5 archive for final review.

For Windows executable packaging, see the dedicated build guide in ``building``. That guide documents the per-app minimal package sets and the recommended small-environment workflow for producing smaller ``.exe`` files.

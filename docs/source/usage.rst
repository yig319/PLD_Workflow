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
   python -m pip install -r requirements-dev.txt

Run the applications
--------------------

.. code-block:: bash

   pld-parameter-form
   pld-raw-visualizer
   python -m pld_workflow

About ``from pld_workflow.app import main``
-------------------------------------------

That import works when the package is installed normally, installed in editable mode, or when a script manually adds ``src/`` to ``sys.path`` first.

The example launcher ``examples/pld_app_parameter.py`` uses the direct source-tree approach, so it can run from a repository checkout without an installation step.

"""Launcher script for the standalone plume manager.

Usage
-----
Run from repository root:

    python examples/plume_manager.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pld_workflow.plume_manager_app import main

if __name__ == "__main__":
    raise SystemExit(main())

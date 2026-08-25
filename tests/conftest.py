from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["QT_QPA_PLATFORM"] = "offscreen"


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

"""PLD workflow tools and desktop parameter form."""

import sys

if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

from .form import GenerateForm

try:
    __version__ = version("pldflow")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = ["GenerateForm", "__version__"]

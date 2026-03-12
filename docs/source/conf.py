import os
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version

sys.path.insert(0, os.path.abspath("../../src"))

project = "pldflow"
copyright = "2026, Yichen Guo"
author = "Yichen Guo"

try:
    release = pkg_version("pldflow")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_title = "pldflow documentation"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

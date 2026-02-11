import os
import sys

# Make project root importable
sys.path.insert(0, os.path.abspath(".."))

project = "SunXCorr"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_member_order = "bysource"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

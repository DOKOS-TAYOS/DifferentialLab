"""Sphinx configuration for DifferentialLab documentation."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# -- Project information -----------------------------------------------------

project = "DifferentialLab"
author = "Alejandro Mata Ali"
copyright = "2026, Alejandro Mata Ali"  # noqa: A001
release = "0.2.1"
version = "0.2"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Enable autosummary for :autosummary: directives (used in API index)
autosummary_generate = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST-Parser settings ----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
]
myst_heading_anchors = 3

# -- Napoleon settings (Google-style docstrings) -----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

suppress_warnings = ["ref.python", "myst.xref_missing"]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/DifferentialLab_logo.png"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": False,
    "includehidden": True,
}
templates_path = ["_templates"]

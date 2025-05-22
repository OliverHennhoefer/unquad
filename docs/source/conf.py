# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "unquad"
copyright = "2025, Oliver Hennhöfer"
author = "Oliver Hennhöfer"
release = "0.1.9"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Options for sphinx-autoapi ------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../unquad"]
autoapi_root = "api"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "show-inheritance": True,
}

autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_root = "api"
autoapi_keep_files = False

master_doc = "index"

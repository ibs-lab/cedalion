# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cedalion"
copyright = "2024, the cedalion developers"
author = "the cedalion developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# fix a margin problem with the rendering of xarray representations in notebooks when
# using the RTD theme
html_css_files = [
    "css/rtd_fixes.css",
]

# -- Configure MyST -----------------------------------------------------------

myst_enable_extensions = [
    "substitution",
]

myst_heading_anchors = 2

# -- Configure sphinxcontrib-bibtex -------------------------------------------

bibtex_bibfiles = ['references.bib']


# -- Substitutions ------------------------------------------------------------

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .strip()
    .decode("ascii")
)

myst_substitutions = {
    "docs_url": "https://doc.ml.tu-berlin.de/cedalion",
    "commit_hash": commit_hash,
}

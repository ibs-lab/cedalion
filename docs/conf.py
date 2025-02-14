# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
from urllib.parse import quote

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
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.linkcode"
    #"autoapi.extension"
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

branch = (
    subprocess.check_output(["git", "branch", "--show-current"])
    .strip()
    .decode("ascii")
)

myst_substitutions = {
    "docs_url": "https://doc.ibs.tu-berlin.de/cedalion/doc/dev",
    "commit_hash": commit_hash,
}

# -- sphinx_autodoc_typehints -------------------------------------------------
always_use_bars_union = True
# specifiying a maximum line length will create line breaks in functions signatures
# and make them easier to read
maximum_signature_line_length = 88

autodoc_type_aliases = {
    "NDTimeSeries" : "cdt.NDTimeSeries",
    "cdt.NDTimeSeries" : "cdt.NDTimeSeries",
    "LabeledPointCloud" : "cdt.LabeledPointCloud",
    "cdt.LabeledPointCloud" : "cdt.LabeledPointCloud",
    "cedalion.Quantity" : "Quantity",
    "pint.Quantity" : "Quantity",
    "Quantity" : "Quantity",
    "ArrayLike" : "ArrayLike",
    "collections.OrderedDict" : "OrderedDict",
}

modindex_common_prefix = ["cedalion."]



# -- sphinx_autoapi_-----------------------------------------------------------
# using autosummary with customized templates as decribed in
# https://github.com/sphinx-doc/sphinx/issues/7912

"""autoapi_dirs = ["../src"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "inherited-members",
    "no-signatures"
]
autoapi_add_toctree_entry = False"""

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"  # Keep member order as in the source code


# -- Nbsphinx gallery ----------------------------------------------------------------
nbsphinx_thumbnails = {
    'examples/*/*': '_static/IBS_Logo_sm.png',
}


# -- linkcode ------- ----------------------------------------------------------------
# adopted from: https://stackoverflow.com/a/75279988
# maybe also incorporate for direct links to line numbers:
# https://github.com/sphinx-doc/sphinx/issues/1556#issuecomment-101027317

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = quote(info['module'].replace('.', '/'))
    if not filename.startswith("tests"):
        filename = "src/" + filename
    if "fullname" in info:
        anchor = info["fullname"]
        anchor = "#:~:text=" + quote(anchor.split(".")[-1])
    else:
        anchor = ""

    # github
    result = f"https://github.com/ibs-lab/cedalion/blob/{branch}/{filename}.py{anchor}"
    # print(result)
    return result

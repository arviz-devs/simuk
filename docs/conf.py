# ruff: noqa
# -*- coding: utf-8 -*-
import os
import sys

from simuk import __version__

sys.path.insert(0, os.path.abspath("../simuk"))

# -- Project information -----------------------------------------------------

project = "Simuk"
author = "ArviZ contributors"
copyright = f"2025, {author}"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "numpydoc",
    "jupyter_sphinx",
]

# -- Extension configuration -------------------------------------------------
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
autosummary_generate = True
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb"}

master_doc = "index"
language = "en"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"
html_css_files = ["custom.css"]
html_title = "Simuk"
html_short_title = "Simuk"
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "collapse_navigation": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "search_bar_text": "Search the docs...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/arviz-devs/simuk",
            "icon": "fa-brands fa-github",
        },
    ],
    # "logo": {
    #     "image_light": "Simuk_flat.png",
    #     "image_dark": "Simuk_flat_white.png",
    # },
}
html_context = {
    "github_user": "arviz-devs",
    "github_repo": "simuk",
    "github_version": "main",
    "doc_path": "docs/",
    "default_mode": "light",
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "simukdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (master_doc, "simuk.tex", "simuk Documentation", "The developers of simuk", "manual"),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "simuk", "simuk Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "simuk",
        "simuk Documentation",
        author,
        "simuk",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

epub_exclude_files = ["search.html"]

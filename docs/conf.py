"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------

project = "CellSem LLM Client"
copyright = "2025, Cellular Semantics"
author = "Cellular Semantics"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # MyST markdown support
    "autoapi.extension",  # API doc generation
    "sphinx.ext.napoleon",  # Google/NumPy docstring style
    "sphinx.ext.viewcode",  # Source code links
    "sphinx_autodoc_typehints",  # Type hints in docs
    "sphinx_design",  # Web components (grids, tabs, cards)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST configuration -----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: syntax for directives
    "deflist",  # Definition lists
    "tasklist",  # Task lists (- [ ] and - [x])
    "attrs_inline",  # Inline attributes {.class #id key=value}
]

# -- AutoAPI configuration --------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/cellsem_llm_client"]
autoapi_root = "api"
autoapi_add_toctree_entry = True  # Let AutoAPI add itself to the toctree
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "class"  # Only include class docstring, not __init__
autoapi_member_order = "groupwise"
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]
# Reduce duplication in generated docs
autoapi_keep_files = False
# Skip documenting Pydantic model fields to reduce duplicates
autoapi_ignore = [
    "*/migrations/*",
    "*/__pycache__/*",
]


def autoapi_skip_member(app, what, name, obj, skip, options):  # type: ignore[no-untyped-def]
    """Custom AutoAPI skip function to reduce Pydantic field duplication."""
    # Skip private members and pydantic internals
    if name.startswith("_"):
        return True
    # Skip Pydantic model fields that cause duplication
    if what == "attribute" and hasattr(obj, "__annotations__"):
        return True
    return skip


def setup(app):  # type: ignore[no-untyped-def]
    """Sphinx setup function."""
    app.connect("autoapi-skip-member", autoapi_skip_member)


# -- HTML output configuration ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_title = f"{project} v{release}"
html_short_title = project

# -- Napoleon configuration (for Google-style docstrings) ------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc typehints configuration ----------------------------------------

typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

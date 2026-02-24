"""Sphinx configuration for proteus documentation."""

import sys
from unittest.mock import MagicMock

# mock compiled CUDA extensions so sphinx can import the package without GPU
_cuda_mocks = [
    "proteus.model.tokenizer.wave_func_tokenizer.learn_aa._C",
    "proteus.model.tokenizer.wave_func_tokenizer.static_aa._C",
]
for mod in _cuda_mocks:
    sys.modules[mod] = MagicMock()

# -- project info -------------------------------------------------------

project = "proteus"
author = "Jaime Cardenas"
release = "0.0.1"

# -- general config ------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- autodoc / autosummary -----------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# mock imports that require GPU-only packages
autodoc_mock_imports = ["flash_attn", "nvidia_ml_py"]

# -- napoleon ------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx ----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- html output ----------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# -- myst-parser -----------------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

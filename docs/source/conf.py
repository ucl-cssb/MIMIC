# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import sys
import os
# Add the directory containing generate_examples_rst.py to Python's path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)
import generate_examples_rst

project = 'MIMIC'
copyright = '2024, Pedro Fontanarrosa'
author = 'Pedro Fontanarrosa'
release = '0.1.0'

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    # Add a link to the source code for classes, functions, etc.
    'sphinx.ext.viewcode',
    'nbsphinx',  # This is the extension for Jupyter notebooks
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# The master toctree document.
master_doc = 'index'

# -- Options for nbsphinx -----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.6/

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)

nbsphinx_execute = 'never'

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# This value selects what content will be inserted into the main body of an autoclass directive.
# The possible values are:
# "class" Only the class’ docstring is inserted. This is the default.
# "both" Both the class’ and the __init__ method’s docstring are concatenated and inserted.
# "init" Only the __init__ method’s docstring is inserted.
autoclass_content = 'both'

# Run the script to generate examples.rst
# This script will generate examples.rst from the jupyter notebooks in the examples directory
generate_examples_rst.main()

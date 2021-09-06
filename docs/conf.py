# Configuration file for the Sphinx documentation builder.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'commplax'
copyright = '2021, The Commplax authors'
author = 'Remi Qirui FAN'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_panels',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
]

# Experimental feature to preserve the default argument values of
# functions in source code and keep them not evaluated for readability:
# autodoc_docstring_signature = True
autodoc_preserve_defaults = True

suppress_warnings = [
    'ref.citation',  # Many duplicated citations in numpy/scipy docstrings.
    'ref.footnote',  # Many unreferenced footnotes in numpy/scipy docstrings
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']

# The main toctree document.
main_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


autosummary_generate = True
master_doc = 'index'

# Since version 3.0 one can set it to 'description' which shows typehints as content
# of function or method and removes them from signature.
autodoc_typehints = 'description'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_book_theme'  # has outdated sphinx version (sphinx 3.X)
html_style = 'css/commplax_theme.css'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for myst ----------------------------------------------
jupyter_execute_notebooks = "force"
execution_allow_errors = False
execution_fail_on_error = True  # Requires https://github.com/executablebooks/MyST-NB/pull/296

# Notebook cell execution timeout; defaults to 30.
execution_timeout = 100


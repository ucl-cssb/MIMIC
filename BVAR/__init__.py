# mypackage/__init__.py

# Import the simulate_var function from the simulation module
from .VARsim import *

# Import the VAR class from the VAR module
from .VARinfer import *

from .utils import *

# You can also include some package-level documentation
__doc__ = """
mypackage: A Python package for simulating VAR models.

This package provides tools and functions to simulate Vector Autoregression (VAR) models.
"""

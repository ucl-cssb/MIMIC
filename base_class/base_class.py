import json
import numpy as np
from BVAR.VARsim import *
from BVAR.VARinfer import *
from gMLV.gMLV_sim import *
from gMLV.gMLV_ML import *


class BaseClass:
    def __init__(self):
        self.data = None

    def simulate(self, sim_type, *args, **kwargs):
        if sim_type == "VAR":
            return VARSimulator(*args, **kwargs)
        elif sim_type == "gMLV":
            return gMLV_sim(*args, **kwargs)
        else:
            raise ValueError("Unknown model type")

    def infer(self, infer_type, *args, **kwargs):
        if infer_type == "VAR":
            return VARInfer(*args, **kwargs)
        elif infer_type == "gMLV":
            return gMLV_sim(*args, **kwargs)  # FIXME: change to gMLV_ML class
        else:
            raise ValueError("Unknown inference type")

    def read_parameters(self, filepath):
        """Read parameters from a JSON file."""
        with open(filepath, 'r') as file:
            parameters = json.load(file)
        return parameters

    def save_data(self, filename, data=None):
        """Save data to a CSV file."""
        if data is None:
            data = self.data
        if data is not None:
            np.savetxt(filename, data, delimiter=",")
        else:
            print("No data to save.")

    def load_data(self, filename):
        """Load data from a CSV file."""
        self.data = np.loadtxt(filename, delimiter=",")

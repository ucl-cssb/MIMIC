from abc import ABC, abstractmethod
import json
import numpy as np
import os
from model_simulate.VARsim import *
from model_infer.VARinfer import *
from model_simulate.gMLV_sim import *
from gMLV.gMLV_ML import *


class BaseModel(ABC):
    def __init__(self):
        self.data = None
        self.model = None
        self.inference = None
        self.parameters = None

    @abstractmethod
    def simulate(self, sim_type, *args, **kwargs):
        if sim_type == "VAR":
            return VARSimulator(*args, **kwargs)
        elif sim_type == "gMLV":
            return gMLV_sim(*args, **kwargs)
        else:
            raise ValueError("Unknown model type")

    @abstractmethod
    def infer(self, infer_type, *args, **kwargs):
        if infer_type == "VAR":
            return VARInfer(*args, **kwargs)
        elif infer_type == "gMLV":
            return gMLV_sim(*args, **kwargs)  # FIXME: change to gMLV_ML class
        else:
            raise ValueError("Unknown inference type")

    @abstractmethod
    def read_parameters(self, filepath):
        """Read parameters from a JSON file."""
        # Check if the filepath ends with .json
        if not filepath.lower().endswith('.json'):
            raise ValueError("Filepath must point to a .json file")

        with open(filepath, 'r') as file:
            parameters = json.load(file)
            self.parameters = parameters

    @abstractmethod
    def save_parameters(self, filepath, parameters=None):
        """Save parameters to a JSON file."""
        if parameters is None:
            parameters = self.parameters
        if parameters is not None:
            # Check if the filename ends with .json, if not append .json
            if not filepath.endswith('.json'):
                filepath += '.json'
            with open(filepath, 'w') as file:
                json.dump(parameters, file)
        else:
            print("No parameters to save.")

    @abstractmethod
    def print_parameters(self):
        """Print parameters to the console."""
        if self.parameters is not None:
            print(json.dumps(self.parameters, indent=4))
        else:
            print("No parameters to print.")

    @abstractmethod
    def save_data(self, filename, data=None):
        """Save data to a CSV file."""
        if data is None:
            data = self.data
        if data is not None:
            # Check if the filename ends with .csv, if not append .csv
            if not filename.endswith('.csv'):
                filename += '.csv'
            np.savetxt(filename, data, delimiter=",")
        else:
            print("No data to save.")

    @abstractmethod
    def load_data(self, filename):
        """Load data from a CSV file."""
        file_extension = os.path.splitext(filename)[1]
        if file_extension.lower() != ".csv":
            raise ValueError("File is not a CSV file.")
        self.data = np.loadtxt(filename, delimiter=",")

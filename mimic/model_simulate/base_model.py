from abc import ABC, abstractmethod
import json
import numpy as np
import os
from model_simulate.VAR_sim import *
from model_simulate.gMLV_sim import *
from model_infer.gMLV_infer import *
from model_infer.VAR_infer import *


class BaseModel(ABC):
    def __init__(self):
        self.data = None
        self.model = None
        self.inference = None
        self.parameters = None

    # check if params are set, else print a warning and use the default values for each simulation type
    def check_params(self, params, sim_type):
        if params is None:
            print(
                f"Warning: No parameters provided for {sim_type} simulation. Using default values.")
            if sim_type == "VAR":
                params = {"n_obs": 100, "coefficients": 2,
                          "initial_values": 2, "noise_stddev": 1, "output": "show"}
            elif sim_type == "gMLV":
                params = {"n": 100, "p": 2, "k": 2, "sigma": 1}
        print(
            f"Using the following parameters for {sim_type} simulation: {params}")
        # IDEA: See if all expected parameters are set, else use default and show a message either here or inside each funciton.
        return params

    @abstractmethod
    def simulate(self, sim_type, params):
        params = self.check_params(params, sim_type)
        if sim_type == "VAR":
            return VARSimulator(**params)
        elif sim_type == "gMLV":
            return gMLV_sim(**params)
        else:
            raise ValueError("Unknown model type")

    # @abstractmethod #NOTE: This is not needed here, since we are going to use it in the infer base class
    # def infer(self, infer_type, *args, **kwargs):
    #     if infer_type == "VAR":
    #         return VARInfer(*args, **kwargs)
    #     elif infer_type == "gMLV":
    #         return gMLV_sim(*args, **kwargs)
    #     else:
    #         raise ValueError("Unknown inference type")

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

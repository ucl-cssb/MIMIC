from abc import ABC, abstractmethod
import os
import json
import pandas as pd
from mimic.model_simulate.VAR_sim import VARSimulator
from mimic.model_simulate.gMLV_sim import gMLV_sim


class BaseModel(ABC):
    """
    Abstract base class for a base model.

    Attributes:
        data: The data generated by the model.
        model: The model used for simulation.
        parameters: The parameters of the model.

    Methods:
        check_params: Checks if parameters are set and uses default values if not.
        simulate: Simulates data based on the specified simulation type and parameters.
        read_parameters: Reads parameters from a JSON file.
        save_parameters: Saves parameters to a JSON file.
        print_parameters: Prints parameters to the console.
        save_data: Saves data to a CSV file.
        load_data: Loads data from a CSV file.
    """

    def __init__(self):
        """
        Initialize a new instance of the BaseModel class.
        """
        self.data = None
        self.model = None
        # self.inference = None #NOTE: This is not needed here, since we are going to use it in the infer base class
        self.parameters = None

    # check if params are set, else print a warning and use the default values for each simulation type
    def check_params(self, params, sim_type):
        """
        Check if parameters are provided for the simulation. If not, use default values.

        This method checks if the `params` dictionary is provided. If not, it prints a warning and sets default values based on the simulation type (`sim_type`). The default values for "VAR" are {"n_obs": 100, "coefficients": 2, "initial_values": 2, "noise_stddev": 1, "output": "show"} and for "gMLV" are {"n": 100, "p": 2, "k": 2, "sigma": 1}.

        Parameters:
        params (dict): The parameters for the simulation. If None, default values are used.
        sim_type (str): The type of simulation. Must be either "VAR" or "gMLV".

        Returns:
        dict: The parameters to be used for the simulation.

        Raises:
        ValueError: If `sim_type` is not "VAR" or "gMLV".
        """
        if params is None:
            print(
                f"Warning: No parameters provided for {sim_type} simulation. Using default values.")
            if sim_type == "VAR":
                params = {"n_obs": 100, "coefficients": [[0.8, -0.2], [0.3, 0.5]],
                          "initial_values": [[1], [2]], "noise_stddev": 1, "output": "show"}
            elif sim_type == "gMLV":
                params = {"n": 100, "p": 2, "k": 2, "sigma": 1}
        print(
            f"Using the following parameters for {sim_type} simulation: {params}")
        # IDEA: See if all expected parameters are set, else use default and show a message either here or inside each funciton.
        return params

    @abstractmethod
    def simulate(self, sim_type, params):
        """
        Simulate data using the specified model and parameters.

        Parameters:
        sim_type (str): The type of simulation. Must be either "VAR" or "gMLV".
        params (dict): The parameters for the simulation.

        Returns:
        The result of the simulation.

        Raises:
        ValueError: If `sim_type` is not "VAR" or "gMLV".
        """
        params = self.check_params(params, sim_type)
        if sim_type == "VAR":
            self.model = VARSimulator(**params)  # create the class instance
            VARSimulator.simulate()  # call the simulate method, this will store data in self.data
            return self.data
        elif sim_type == "gMLV":
            self.model = gMLV_sim(**params)  # create the class instance
            # call the simulate method, this will store data in self.data
            return self.model.simulate()
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

    def read_parameters(self, filepath):
        """
        Read parameters from a JSON file.

        Parameters:
        filepath (str): The path to the JSON file.

        Raises:
        ValueError: If `filepath` does not point to a .json file.
        FileNotFoundError: If `filepath` does not exist.
        """
        # Check if the filepath ends with .json
        if not filepath.lower().endswith('.json'):
            raise ValueError("Filepath must point to a .json file.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        try:
            with open(filepath, 'r') as file:
                self.parameters = json.load(file)
        except Exception as e:
            print(f"Error reading parameters from {filepath}: {e}")
            return False

    def save_parameters(self, filepath, parameters=None):
        """
        Save parameters to a JSON file.

        Parameters:
        filepath (str): The path to the JSON file.
        parameters (dict, optional): The parameters to save. If None, the instance's parameters are used.

        Raises:
        ValueError: If `filepath` does not point to a .json file.
        FileNotFoundError: If the directory to save the file does not exist.
        """
        if not filepath.endswith('.json'):
            raise ValueError("Filepath must point to a .json file.")
        if not os.path.exists(os.path.dirname(filepath)):
            raise FileNotFoundError(
                f"No directory found at {os.path.dirname(filepath)}")

        parameters = parameters if parameters is not None else self.parameters
        if parameters is None:
            print("No parameters to save.")
            return

        try:
            with open(filepath, 'w') as file:
                json.dump(parameters, file)
        except Exception as e:
            print(f"Error saving parameters to {filepath}: {e}")
            return False

    def print_parameters(self):
        """
        Print parameters to the console.

        If the instance's parameters are None, prints "No parameters to print."
        """
        if self.parameters is not None:
            print(json.dumps(self.parameters, indent=4))
        else:
            print("No parameters to print.")

    def save_data(self, filename, data=None):
        """
        Save data to a CSV file.

        Parameters:
        filename (str): The name of the CSV file.
        data (numpy array, optional): The data to save. If None, the instance's data is used.

        Raises:
        ValueError: If `filename` does not end with .csv.
        FileNotFoundError: If the directory to save the file does not exist.
        """
        if not filename.endswith('.csv'):
            raise ValueError("Filename must end with .csv.")
        if not os.path.exists(os.path.dirname(filename)):
            raise FileNotFoundError(
                f"No directory found at {os.path.dirname(filename)}")

        data = data if data is not None else self.data
        if data is None:
            print("No data to save.")
            return

        try:
            pd.DataFrame(data).to_csv(filename, index=False)
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            return False

    def load_data(self, filename):
        """
        Load data from a CSV file.

        Parameters:
        filename (str): The name of the CSV file.

        Raises:
        ValueError: If `filename` does not point to a .csv file.
        FileNotFoundError: If `filename` does not exist.
        """
        if not filename.endswith('.csv'):
            raise ValueError("Filename must point to a .csv file.")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")

        try:
            self.data = pd.read_csv(filename)
        except Exception as e:
            print(f"Error reading data from {filename}: {e}")
            return False

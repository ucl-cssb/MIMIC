import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for creating and managing simulation models.

    This class serves as a foundation for any type of model that requires
    managing data, parameters, and basic I/O operations. It defines a
    common interface for parameter handling, data simulation, and data
    persistence.

    Attributes:
        data (np.ndarray | None):
            Holds the output data generated by the model's simulation.
            This could be None if the model has not yet produced any data.
        model (object | None):
            A generic placeholder for the specific simulation model instance.
            This attribute should be overridden in subclasses with an actual
            model representation.
        parameters (dict | None):
            A dictionary containing the parameters that control the model's
            behavior. Parameters should be defined in subclasses or set through
            the provided methods.

    Abstract Methods:
        set_parameters(self):
            Should be implemented by subclasses to define how model parameters
            are set or updated.
        simulate(self):
            Should be implemented by subclasses to define the model's simulation
            process based on the set parameters.

    Methods:
        check_params(self, params, sim_type):
            Checks provided parameters against required ones for a given
            simulation type, applying default values if necessary.
        read_parameters(self, filepath):
            Reads model parameters from a specified JSON file and updates
            the model's parameters accordingly.
        save_parameters(self, filepath, parameters=None):
            Saves the model's current parameters to a JSON file. Optionally,
            a specific set of parameters can be provided to save instead.
        print_parameters(self, precision=2):
            Prints the current set of model parameters to the console, formatting
            numpy arrays with specified precision.
        save_data(self, filename, data=None):
            Saves the model's generated data to a CSV file. Optionally, specific
            data can be provided to save instead.
        load_data(self, filename):
            Loads data from a specified CSV file into the model's `data` attribute.
        _custom_array_to_string(self, array, precision=2):
            Converts a numpy array to a string representation with specified precision.
        update_attributes(self):
            Updates class attributes based on the current parameters dictionary.
    """

    def __init__(self, debug: Optional[str] = None):
        """
        Initializes the BaseModel with default values.
        """
        self.data: Optional[np.ndarray] = None
        self.model: Optional[object] = None
        self._debug = None  # Initialize the private attribute
        self.debug = debug  # Set the debug level through the property
        self.parameters: Optional[Dict[str, Union[int,
                                                  float, None, np.ndarray, str, Any]]] = None

    # The debug property is a getter and setter for the private attribute
    # _debug.
    @property
    def debug(self) -> Optional[str]:
        """Gets the current debug level."""
        return self._debug

    # The setter for the debug property only allows setting the debug level to
    # None, 'low', or 'high'.
    @debug.setter
    def debug(self, value: Optional[str]) -> None:
        """Sets the debug level, allowing only None, 'low', or 'high'."""
        if value not in {None, "low", "high"}:
            raise ValueError("Debug level must be None, 'low', or 'high'.")
        self._debug = value

    # Example usage of the debug property:
    # model = BaseModel(debug="low")  # Valid
    # model.debug = "high"            # Valid
    # model.debug = "invalid"         # Raises ValueError

    # check if params are set, else print a warning and use the default values
    # for each simulation type

    @abstractmethod
    def set_parameters(self) -> None:
        """
        Abstract method for setting parameters specific to the model.

        Subclasses should implement this method to customize parameter handling.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def update_attributes(self) -> None:
        """
        Updates class attributes based on the current parameters dictionary.

        This method iterates through the `parameters` dictionary and updates
        the class attributes with corresponding keys to match the parameter values.
        """
        if self.parameters:
            for key, value in self.parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def read_parameters(self, filepath: str) -> None:
        """
        Reads parameters from a JSON file and updates the model's parameters.

        Parameters that can be converted to numpy arrays are handled accordingly.

        Parameters:
            filepath (str): Path to the JSON file containing parameters.

        Raises:
            ValueError: If the filepath does not point to a .json file.
            FileNotFoundError: If the specified file does not exist.
        """
        if not filepath.lower().endswith('.json'):
            raise ValueError("Filepath must point to a .json file.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                loaded_params = json.load(file)
            # Convert lists back to numpy arrays where appropriate
            self.parameters = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in loaded_params.items()
            }
            # Update class attributes based on the newly loaded parameters
            self.update_attributes()
        except Exception as e:
            print(f"Error reading parameters from {filepath}: {e}")

    def check_params(
            self, params: Optional[Dict[str, Any]], sim_type: str) -> None:
        """
        Checks and updates simulation parameters with defaults if necessary.

        Validates provided parameters against the simulation type and updates missing
        or `None` values with defaults. Warnings are issued for missing parameters.

        Parameters:
            params (dict): User-provided parameters for the simulation.
            sim_type (str): The type of simulation to perform ('VAR', 'gMLV', or 'sVAR').

        Raises:
            ValueError: If an unsupported simulation type is specified.
        """
        # NOTE: This method is here instead of being implemented by each derived class to avoid code repetition. It is used to check if parameters are set and use default values if not.
        # sourcery skip: use-named-expression
        # Define default parameters for each simulation type
        # FIXME: #46 Change the default parameter generation to whatever
        # dimension the user wants. Right now it's hardcoded to 2.
        default_params_VAR = {"n_obs": 100, "coefficients": [[0.8, -0.2], [0.3, 0.5]],
                              "initial_values": [[1], [2]], "noise_stddev": 1, "output": "show"}
        default_params_gMLV: dict[str,
                                  object] = {"num_species": 2,
                                             "num_metabolites": 0,
                                             "num_perturbations": 0,
                                             "mu": None,
                                             "M": None,
                                             "beta": None,
                                             "epsilon": None}
        default_params_gLV: dict[str, object] = {
            "num_species": 2, "mu": None, "M": None, "epsilon": None}

        # TODO: #48 Do this programmatically (see how the class is initiated
        # and use the same logic to generate the default parameters for the
        # simulation type.)
        default_params_sVAR = {"n_obs": 100, "coefficients": [[0.8, -0.2], [0.3, 0.5]], "initial_values": [[1], [2]], "noise_stddev": 1.0,
                               "output": "show", "coefficientsM": [[0.0, -0.5, 0.0], [0.1, 0.1, -0.1], [-0.2, 0.1, 0.3]], "initial_valuesM": [[2], [0], [0]]}

        # Determine default parameters based on simulation type
        if sim_type == "VAR":
            default_params = default_params_VAR
        elif sim_type == "sVAR":
            default_params = default_params_sVAR
        elif sim_type == "gMLV":
            default_params = default_params_gMLV
        elif sim_type == "gLV":
            default_params = default_params_gLV
        else:
            raise ValueError("sim_type must be 'VAR' or 'gMLV'.")

        # Check if no parameters were provided and warn the user
        if params is None:
            if self.debug in ["low", "high"]:
                print(
                    f"Warning: No parameters provided for {sim_type} simulation. Using default values.")
        else:
            # Identify missing or None parameters
            missing_params = [
                key for key in default_params if key not in params or params[key] is None]
            if missing_params:
                print(
                    f"Warning: Missing or None parameters for {sim_type} simulation. Using default values for: {missing_params}")
            # Update the default parameters with the provided ones
            for key, value in params.items():
                if value is not None:
                    default_params[key] = value

        if self.debug == "high":
            print(
                f"Using the following parameters for {sim_type} simulation: {default_params}")
        self.parameters = default_params
        self.update_attributes()

    def _custom_array_to_string(
            self,
            array: np.ndarray,
            precision: int = 2) -> str:
        """
        Converts a numpy array to a string representation with specified precision.

        Parameters:
            array (numpy.ndarray): The numpy array to convert.
            precision (int): The number of decimal places for floating-point numbers.

        Returns:
            str: A string representation of the numpy array.
        """
        with np.printoptions(precision=precision, suppress=True):
            return np.array2string(array, separator=' ')

    def print_parameters(self, precision: int = 2) -> None:
        """
        Prints the model's parameters to the console with formatted numpy arrays.

        Numpy arrays are converted to strings with compact formatting, using the
        specified precision for floating-point numbers.

        Parameters:
            precision (int): Precision for formatting numpy array elements.
        """
        # Check the class-level debug level
        if self.debug in ["low", "high"]:
            print("Model parameters:")
            print(f"Model: {self.model}")
            if self.parameters is not None:
                parameters = {
                    k: self._custom_array_to_string(
                        v, precision) if isinstance(v, np.ndarray) else v
                    for k, v in self.parameters.items()
                }
                for param, value in parameters.items():
                    print(f"{param}: {value}")
            else:
                print("No parameters to print.")

    def save_parameters(self,
                        filepath: str,
                        parameters: Optional[Dict[str,
                                                  Union[int,
                                                        float,
                                                        np.ndarray,
                                                        str,
                                                        Any]]] = None) -> None:
        """
        Saves the model's parameters to a JSON file.

        Parameters can be optionally provided; otherwise, the model's current parameters are used.
        Numpy arrays are converted to lists for JSON serialization.

        Parameters:
            filepath (str): Path to the JSON file for saving parameters.
            parameters (dict, optional): Parameters to save. Uses model's parameters if None.

        Raises:
            ValueError: If the filepath does not end with '.json'.
            FileNotFoundError: If the specified directory does not exist.
        """
        if not filepath.endswith('.json'):
            raise ValueError("Filepath must point to a .json file.")
        if not os.path.exists(os.path.dirname(
                filepath)) and os.path.dirname(filepath) != '':
            raise FileNotFoundError(
                f"No directory found at {os.path.dirname(filepath)}")

        parameters = parameters if parameters is not None else self.parameters
        if parameters is None:
            print("No parameters to save.")
            return

        # Prepare parameters for JSON serialization
        # Convert numpy arrays to lists
        serializable_params = {
            k: v.tolist() if isinstance(v, np.ndarray) else v  # pylint: disable=E1101
            for k, v in parameters.items()
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(serializable_params, file)
        except Exception as e:
            print(f"Error saving parameters to {filepath}: {e}")

    def save_data(self, filename: str,
                  data: Optional[np.ndarray] = None) -> None:
        """
        Saves data to a CSV file.

        If data is not provided, the model's current data is used. Raises exceptions
        if the filename does not end with '.csv' or if the specified directory does not exist.

        Parameters:
            filename (str): The name of the CSV file to save data.
            data (numpy.ndarray, optional): The data to save. Uses model's data if None.

        Raises:
            ValueError: If `filename` does not end with '.csv'.
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
            pd.DataFrame(data).to_csv(filename, index=False, header=False)
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")

    def load_data(self, filename: str) -> None:
        """
        Loads data from a CSV file into the model's data attribute.

        Parameters:
            filename (str): The name of the CSV file from which to load data.

        Raises:
            ValueError: If `filename` does not point to a .csv file.
            FileNotFoundError: If `filename` does not exist.
        """
        if not filename.endswith('.csv'):
            raise ValueError("Filename must point to a .csv file.")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")

        try:
            self.data = pd.read_csv(filename, header=None).values.tolist()
        except Exception as e:
            print(f"Error reading data from {filename}: {e}")

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """
        Abstract method to simulate data based on the model's parameters.

        Subclasses must implement this method to provide specific simulation functionality.
        """
        raise NotImplementedError("Subclasses must implement this method")

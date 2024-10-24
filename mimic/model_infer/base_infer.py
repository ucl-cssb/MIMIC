from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union
import numpy as np
import pandas as pd


class BaseInfer(ABC):
    """
    Abstract base class for inference models.

    This class provides a structure for setting priors, managing data, and handling
    the inference process. Specific inference methods should be implemented in subclasses.

    Attributes:
        priors (dict): Dictionary holding prior distributions for model parameters.
        data (np.ndarray | None): The data on which inference is performed.
        dataS (np.ndarray | None): The secondary data (e.g., metabolite data).
        model (object | None): Placeholder for the specific model used in inference.
        results (Any | None): Results of the inference process.
    """

    def __init__(self, debug: Optional[str] = None):
        # Initialize the debug level with a default of None
        self._debug = None
        self.debug = debug  # Set the debug level through the property
        # Initialize priors as an empty dictionary
        self.priors: Dict[str, Any] = {}
        # Other initializations
        self.data: Optional[np.ndarray] = None
        self.dataS: Optional[np.ndarray] = None
        self.model: Optional[object] = None
        self.results: Optional[Any] = None

    @property
    def debug(self) -> Optional[str]:
        """Gets the current debug level."""
        return self._debug

    @debug.setter
    def debug(self, value: Optional[str]) -> None:
        """Sets the debug level, allowing only None, 'low', or 'high'."""
        if value not in {None, "low", "high"}:
            raise ValueError("Debug level must be None, 'low', or 'high'.")
        self._debug = value

    def _validate_data(self, data):
        """
        Validates and converts the input data to a numpy array.

        Parameters:
            data: The data to validate and convert. Can be a DataFrame, ndarray, list, or tuple.

        Returns:
            np.ndarray: The validated and converted data.
        """
        if data is None:
            return None
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            try:
                return np.array(data)
            except Exception as e:
                raise TypeError(
                    f"Data could not be converted to a numpy array: {e}") from e
        else:
            raise TypeError(
                "Unsupported data type. Data must be a DataFrame, ndarray, list, or tuple.")

    def import_data(
            self,
            file_path: str,
            index_col: Optional[int] = None,
            parse_dates: bool = False,
            data_type: str = 'X') -> None:
        """
        Imports data from a .csv file and assigns it to the appropriate attribute.

        Parameters:
            file_path (str): The path to the .csv file.
            index_col (int, optional): Column to use as the row labels of the DataFrame.
            parse_dates (bool, optional): Parse dates as datetime.
            data_type (str): Specify whether the data is for 'X' (abundance) or 'S' (metabolite).

        Raises:
            ValueError: If there is an issue with importing or validating the data.
        """
        try:
            data = pd.read_csv(file_path, index_col=index_col,
                               parse_dates=parse_dates)
            if data_type == 'X':
                self.data = self._validate_data(data)
            elif data_type == 'S':
                self.dataS = self._validate_data(data)
            else:
                raise ValueError("data_type must be either 'X' or 'S'.")
        except Exception as e:
            raise ValueError(
                f"Failed to import data from {file_path}: {e}") from e

    def set_priors(self, priors: Dict[str, Any]) -> None:
        """
        Sets or updates the priors for the inference model.

        Parameters:
            priors (dict): A dictionary where keys are parameter names and values are the prior distributions.
        """
        self.priors.update(priors)  # Update existing priors with new values

    @abstractmethod
    def run_inference(self, **kwargs) -> None:
        """
        Abstract method to run the inference process.

        This method should be implemented by subclasses to define how inference is conducted.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def save_results(self, filepath: str) -> None:
        """
        Saves the results of the inference process to a file.

        Parameters:
            filepath (str): The path to save the results.
        """
        if self.results is not None:
            np.save(filepath, self.results)
        else:
            print("No results to save.")

    def load_results(self, filepath: str) -> None:
        """
        Loads previously saved inference results from a file.

        Parameters:
            filepath (str): The path from where to load the results.
        """
        self.results = np.load(filepath, allow_pickle=True)

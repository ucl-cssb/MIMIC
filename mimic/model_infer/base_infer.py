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
        model (object | None): Placeholder for the specific model used in inference.
        results (Any | None): Results of the inference process.
    """

    def __init__(self):
        self.priors: Optional[Dict[str, Any]] = None
        self.data: Optional[np.ndarray] = None
        self.model: Optional[object] = None
        self.results: Optional[Any] = None

    def set_priors(self, priors: Dict[str, Any]) -> None:
        """
        Sets or updates the priors for the inference model.

        Parameters:
            priors (dict): A dictionary where keys are parameter names and values are the prior distributions.
        """
        self.priors = priors

    def import_data(self, data: Union[str, pd.DataFrame, np.ndarray], data_type: str = 'X') -> None:
        """
        Imports data for the inference process.

        Parameters:
            data (str | pd.DataFrame | np.ndarray): Path to a CSV file, a pandas DataFrame, or a numpy array.
            data_type (str): Type of data ('X' for main data, 'S' for secondary data, etc.).
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data).values
        elif isinstance(data, pd.DataFrame):
            self.data = data.values
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(
                "Unsupported data format. Provide a CSV path, DataFrame, or ndarray.")

    @abstractmethod
    def run_inference(self, **kwargs) -> None:
        """
        Abstract method to run the inference process.

        This method should be implemented by subclasses to define how inference is conducted.
        """
        pass

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

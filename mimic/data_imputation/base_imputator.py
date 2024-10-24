# mimic/data_imputation/base_imputer.py

import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseImputer(ABC):
    """
    Abstract base class for imputators. Provides a template for handling data imputation
    with methods to save and load data and a structure for defining imputation methods.
    """

    def __init__(self, debug: Optional[str] = None):
        self.data = None
        self._debug = None
        self.debug = debug  # Set the debug level through the property

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

    @abstractmethod
    def impute_missing_values(
            self,
            dataset: pd.DataFrame,
            feature_columns: list,
            output_columns: list,
            target_column: str,
            kernel: Optional[str] = None) -> pd.DataFrame:
        """
        Abstract method to impute missing values in the dataset.
        Subclasses must implement this method to provide specific imputation functionality.

        :param dataset: The dataset containing missing values.
        :param feature_columns: List of feature columns to use in the imputation.
        :param output_columns: List of columns to store imputed values.
        :param target_column: The target column where missing values are imputed.
        :param kernel: Optional kernel parameter for imputation.
        :return: Dataset with imputed values in the target column.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def save_data(self, filename: str) -> None:
        """
        Saves data to a CSV file.

        :param filename: The filename to save the data to.
        :param data: The DataFrame to save.
        :raises ValueError: If the filename does not end with '.csv'.
        :raises FileNotFoundError: If the directory to save the file does not exist.
        :raises ValueError: If self.data is None.
        """
        if self.data is None:
            raise ValueError("No data to save. self.data is None.")
        if not filename.endswith('.csv'):
            raise ValueError("Filename must end with .csv.")
        if not os.path.exists(os.path.dirname(
                filename)) and os.path.dirname(filename) != '':
            raise FileNotFoundError(
                f"No directory found at {os.path.dirname(filename)}")

        self.data.to_csv(filename, index=False)

    def load_data(self, filename: str) -> None:
        """
        Loads data from a CSV file.

        :param filename: The filename to load data from.
        :return: Loaded data as a DataFrame.
        :raises ValueError: If the filename does not end with '.csv'.
        :raises FileNotFoundError: If the file does not exist.
        """
        if not filename.endswith('.csv'):
            raise ValueError("Filename must point to a .csv file.")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")

        self.data = pd.read_csv(filename)

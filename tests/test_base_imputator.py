# tests/test_base_imputer.py

import pytest
import pandas as pd
import os
from typing import Optional
from mimic.data_imputation.base_imputator import BaseImputer


class ConcreteImputer(BaseImputer):
    def impute_missing_values(
        self,
        dataset: pd.DataFrame,
        feature_columns: list,
        output_columns: list,
        target_column: str,
        kernel: Optional[str] = None
    ) -> pd.DataFrame:
        # Simple implementation for testing
        return dataset.fillna(0)


def test_imputer_initialization():
    imputer = ConcreteImputer()
    assert imputer.data is None


def test_save_data(tmp_path):
    imputer = ConcreteImputer()
    imputer.data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    file_path = os.path.join(tmp_path, 'test_data.csv')
    imputer.save_data(file_path)

    assert os.path.exists(file_path)
    loaded_data = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(imputer.data, loaded_data)


def test_save_data_no_data():
    imputer = ConcreteImputer()
    with pytest.raises(ValueError, match="No data to save. self.data is None."):
        imputer.save_data('dummy.csv')


def test_save_data_invalid_extension():
    imputer = ConcreteImputer()
    imputer.data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    with pytest.raises(ValueError, match="Filename must end with .csv."):
        imputer.save_data('dummy.txt')


def test_load_data(tmp_path):
    imputer = ConcreteImputer()
    file_path = os.path.join(tmp_path, 'test_data.csv')

    pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }).to_csv(file_path, index=False)

    imputer.load_data(file_path)

    expected_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    pd.testing.assert_frame_equal(imputer.data, expected_data)


def test_load_data_invalid_extension():
    imputer = ConcreteImputer()
    with pytest.raises(ValueError, match="Filename must point to a .csv file."):
        imputer.load_data('dummy.txt')


def test_load_data_file_not_found():
    imputer = ConcreteImputer()
    with pytest.raises(FileNotFoundError, match="No file found at dummy.csv"):
        imputer.load_data('dummy.csv')

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from unittest.mock import MagicMock
from mimic.model_infer.base_infer import BaseInfer

# Create a subclass of BaseInfer for testing purposes


class TestInfer(BaseInfer):
    def run_inference(self, **kwargs) -> None:
        # A simple mock implementation for testing
        self.results = "inference results"


@pytest.fixture
def test_infer():
    return TestInfer()


def test_initialization(test_infer):
    # Test initial state of the class
    assert test_infer.priors == {}
    assert test_infer.data is None
    assert test_infer.dataS is None
    assert test_infer.model is None
    assert test_infer.results is None


def test_validate_data(test_infer):
    # Test various data types for validation
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert np.array_equal(test_infer._validate_data(df), df.values)

    array = np.array([[1, 2], [3, 4]])
    assert np.array_equal(test_infer._validate_data(array), array)

    data_list = [[1, 2], [3, 4]]
    assert np.array_equal(test_infer._validate_data(
        data_list), np.array(data_list))

    with pytest.raises(TypeError):
        test_infer._validate_data("invalid data type")


def test_import_data(mocker, test_infer):
    # Mocking pd.read_csv
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame(
        {'A': [1, 2], 'B': [3, 4]}))

    # Test importing data of type 'X'
    test_infer.import_data("dummy_path.csv", data_type='X')
    assert test_infer.data is not None
    assert test_infer.dataS is None

    # Test importing data of type 'S'
    test_infer.import_data("dummy_path.csv", data_type='S')
    assert test_infer.dataS is not None
    assert test_infer.data is not None

    # Test invalid data type
    with pytest.raises(ValueError):
        test_infer.import_data("dummy_path.csv", data_type='invalid')


def test_set_priors(test_infer):
    priors = {"param1": "prior1", "param2": "prior2"}
    test_infer.set_priors(priors)
    assert test_infer.priors == priors

    # Test updating priors
    new_priors = {"param2": "new_prior2", "param3": "prior3"}
    test_infer.set_priors(new_priors)
    assert test_infer.priors == {"param1": "prior1",
                                 "param2": "new_prior2", "param3": "prior3"}


def test_run_inference(test_infer):
    test_infer.run_inference()
    assert test_infer.results == "inference results"


def test_save_results(mocker, test_infer):
    # Mock np.save
    mock_save = mocker.patch("numpy.save")

    test_infer.results = np.array([1, 2, 3])
    test_infer.save_results("dummy_path.npy")

    mock_save.assert_called_once_with("dummy_path.npy", test_infer.results)


def test_load_results(mocker, test_infer):
    # Mock np.load
    mock_load = mocker.patch("numpy.load", return_value=np.array([1, 2, 3]))

    test_infer.load_results("dummy_path.npy")
    assert np.array_equal(test_infer.results, np.array([1, 2, 3]))

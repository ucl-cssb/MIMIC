import pytest
import numpy as np
import os
import json
from unittest.mock import MagicMock
from mimic.model_simulate.base_model import BaseModel


class MockModel(BaseModel):
    """A mock subclass of BaseModel to test its abstract methods."""

    def set_parameters(self):
        """Mock implementation of abstract method."""
        pass

    def simulate(self):
        """Mock implementation of abstract method."""
        pass


@pytest.fixture
def mock_model():
    """Fixture to create a mock model for testing."""
    return MockModel()


def test_initialization(mock_model):
    """Test the initialization of the base model."""
    assert mock_model.data is None
    assert mock_model.model is None
    assert mock_model.parameters is None


def test_set_parameters(mock_model):
    """Test that set_parameters can be called without errors."""
    mock_model.set_parameters()


def test_simulate(mock_model):
    """Test that simulate can be called without errors."""
    mock_model.simulate()


def test_update_attributes(mock_model):
    """Test that update_attributes updates class attributes correctly."""
    mock_model.parameters = {
        "param1": 1,
        "param2": 2.0,
        "param3": np.array([3, 4, 5])
    }
    mock_model.param1 = None
    mock_model.param2 = None
    mock_model.param3 = None

    mock_model.update_attributes()

    assert mock_model.param1 == 1
    assert mock_model.param2 == 2.0

    # Ensure that param3 is not None before using array_equal
    if mock_model.param3 is not None:
        assert np.array_equal(mock_model.param3, np.array([3, 4, 5]))
    else:
        pytest.fail("param3 is None, expected a numpy array")


def test_read_parameters(mock_model, tmpdir):
    """Test reading parameters from a JSON file."""
    filepath = tmpdir.join("params.json")
    params = {"param1": 1, "param2": [1.0, 2.0, 3.0]}
    with open(filepath, "w") as f:
        json.dump(params, f)

    mock_model.read_parameters(str(filepath))

    assert mock_model.parameters["param1"] == 1
    assert np.array_equal(
        mock_model.parameters["param2"], np.array([1.0, 2.0, 3.0]))


def test_check_params(mock_model):
    """Test that check_params assigns default values and updates attributes."""
    params = {"num_species": None}
    mock_model.check_params(params, sim_type="gMLV")

    assert mock_model.parameters["num_species"] == 2  # Default value
    assert mock_model.parameters["num_metabolites"] == 0  # Default value


def test_custom_array_to_string(mock_model):
    """Test conversion of a numpy array to a string with specified precision."""
    array = np.array([1.123456, 2.654321])
    result = mock_model._custom_array_to_string(array, precision=2)

    assert result == "[1.12 2.65]"


def test_print_parameters(mock_model, capsys):
    """Test printing of model parameters."""
    mock_model.model = "TestModel"
    mock_model.parameters = {"param1": 1, "param2": np.array([1.0, 2.0])}

    mock_model.print_parameters(precision=2)

    captured = capsys.readouterr()
    assert "Model: TestModel" in captured.out
    assert "param1: 1" in captured.out
    assert "param2: [1. 2.]" in captured.out


def test_save_parameters(mock_model, tmpdir):
    """Test saving parameters to a JSON file."""
    mock_model.parameters = {"param1": 1, "param2": np.array([1.0, 2.0])}
    filepath = tmpdir.join("params.json")

    mock_model.save_parameters(str(filepath))

    with open(filepath, "r") as f:
        saved_params = json.load(f)

    assert saved_params["param1"] == 1
    assert saved_params["param2"] == [1.0, 2.0]  # Converted to list


def test_save_data(mock_model, tmpdir):
    """Test saving data to a CSV file."""
    mock_model.data = np.array([[1, 2], [3, 4]])
    filepath = tmpdir.join("data.csv")

    mock_model.save_data(str(filepath))

    saved_data = np.loadtxt(filepath, delimiter=",")

    assert np.array_equal(saved_data, mock_model.data)


def test_load_data(mock_model, tmpdir):
    """Test loading data from a CSV file."""
    data = np.array([[1, 2], [3, 4]])
    filepath = tmpdir.join("data.csv")
    np.savetxt(filepath, data, delimiter=",")

    mock_model.load_data(str(filepath))

    assert np.array_equal(np.array(mock_model.data), data)

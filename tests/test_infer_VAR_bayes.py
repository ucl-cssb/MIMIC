import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pymc as pm
import arviz as az
from mimic.model_infer.infer_VAR_bayes import infer_VAR


@pytest.fixture
def example_data():
    return np.random.rand(10, 3)


@pytest.fixture
def example_metabolite_data():
    return np.random.rand(10, 2)


def test_initialization(example_data):
    # Test initialization without data
    model = infer_VAR()
    assert model.data is None
    assert model.dataS is None
    assert model.coefficients is None

    # Test initialization with data
    model = infer_VAR(data=example_data)
    assert model.data is not None
    assert model.data.shape == (10, 3)


def test_run_inference(example_data):
    # Initialize the model with data
    model = infer_VAR(data=example_data)

    # Create a mock trace object that mimics the structure expected by arviz
    mock_trace = MagicMock()
    
    # Mock pymc's sample method to return the mock trace
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
         patch.object(az, 'summary'), patch.object(az, 'plot_posterior'), \
         patch.object(az, 'to_netcdf'):
        model.run_inference(samples=500, tune=200, cores=2)
        mock_sample.assert_called_once_with(500, tune=200, cores=2)


def test_run_inference_large(example_data):
    # Create larger dummy data
    data = np.random.rand(50, 3)

    # Initialize the model with data
    model = infer_VAR(data=data)

    # Create a mock trace object
    mock_trace = MagicMock()

    # Mock pymc's sample method
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
         patch.object(az, 'summary'), patch.object(az, 'plot_posterior'), \
         patch.object(az, 'to_netcdf'):
        model.run_inference_large(samples=1000, tune=500, cores=4)
        mock_sample.assert_called_once_with(1000, tune=500, cores=4)


def test_run_inference_xs(example_data, example_metabolite_data):
    # Initialize the model with data and metabolite data
    model = infer_VAR(data=example_data, dataS=example_metabolite_data)

    # Create a mock trace object
    mock_trace = MagicMock()

    # Mock pymc's sample method
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
         patch.object(az, 'summary'), patch.object(az, 'plot_posterior'), \
         patch.object(az, 'to_netcdf'):
        model.run_inference_xs(samples=500, tune=200, cores=2)
        mock_sample.assert_called_once_with(500, tune=200, cores=2)

    # Test for missing metabolite data
    model.dataS = None
    with pytest.raises(ValueError):
        model.run_inference_xs()


def test_posterior_analysis(mocker, example_data, example_metabolite_data):
    # Mock arviz.from_netcdf to return a mock inference data object
    mock_inference_data = MagicMock()
    mocker.patch('arviz.from_netcdf', return_value=mock_inference_data)

    model = infer_VAR(data=example_data, dataS=example_metabolite_data)

    # Mock plotting and saving methods
    mock_plot_heatmap = mocker.patch.object(model, 'plot_heatmap')

    # Mock np.load to return a context manager with example data
    mock_np_load = MagicMock()
    mock_np_load.__enter__.return_value = {
        'dataX': example_data, 'dataS': example_metabolite_data}
    mocker.patch('numpy.load', return_value=mock_np_load)

    # Ensure that the trace exists in model to avoid loading errors
    model.last_trace = mock_inference_data
    model.last_data = (example_data, example_metabolite_data)

    model.posterior_analysis(netcdf_filename="mock.nc")

    # Check if plot_heatmap was called
    mock_plot_heatmap.assert_called_once()

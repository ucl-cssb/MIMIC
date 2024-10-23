from unittest.mock import MagicMock, patch

import arviz as az
import matplotlib
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from mimic.model_infer.infer_VAR_bayes import infer_VAR

matplotlib.use('Agg')


@pytest.fixture
def example_data():
    return np.random.rand(10, 3)


@pytest.fixture
def example_metabolite_data():
    return np.random.rand(10, 2)


def create_mock_trace(dim, chains=2, draws=500):
    # Mock data for x0 and A variables
    x0_data = np.random.randn(chains, draws, dim)
    A_data = np.random.randn(chains, draws, dim, dim)

    x0 = xr.DataArray(
        x0_data,
        dims=["chain", "draw", "x0_dim"],
        coords={"x0_dim": np.arange(dim)}
    )

    A = xr.DataArray(
        A_data,
        dims=["chain", "draw", "A_dim_0", "A_dim_1"],
        coords={"A_dim_0": np.arange(dim), "A_dim_1": np.arange(dim)}
    )

    posterior = xr.Dataset({"x0": x0, "A": A})

    trace = az.InferenceData(posterior=posterior)
    return trace


def create_mock_trace_xs(nX, nS, chains=2, draws=500):
    Ah_data = np.random.randn(chains, draws, nX, nX)
    Bh_data = np.random.randn(chains, draws, nS, nX)

    Ah = xr.DataArray(
        Ah_data,
        dims=["chain", "draw", "Ah_dim_0", "Ah_dim_1"],
    )

    Bh = xr.DataArray(
        Bh_data,
        dims=["chain", "draw", "Bh_dim_0", "Bh_dim_1"],
    )

    posterior = xr.Dataset({"Ah": Ah, "Bh": Bh})

    trace = az.InferenceData(posterior=posterior)
    return trace


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

    # Determine the dimension based on example_data
    dim = example_data.shape[1]

    # Create a mock trace object with the correct dimensions
    mock_trace = create_mock_trace(dim=dim)

    # Mock pymc's sample method to return the mock trace
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
            patch('mimic.model_infer.infer_VAR_bayes.plt.savefig'), \
            patch('mimic.model_infer.infer_VAR_bayes.az.to_netcdf'), \
            patch('mimic.model_infer.infer_VAR_bayes.np.savez'):
        model.run_inference(samples=500, tune=200, cores=2)
        mock_sample.assert_called_once_with(draws=500, tune=200, cores=2)


def test_run_inference_large(example_data):
    # Create larger dummy data
    data = np.random.rand(50, 3)

    # Initialize the model with data
    model = infer_VAR(data=data)

    dim = data.shape[1]

    # Create a mock trace object
    mock_trace = create_mock_trace(dim=dim)

    # Mock pymc's sample method
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
            patch('mimic.model_infer.infer_VAR_bayes.plt.savefig'), \
            patch('mimic.model_infer.infer_VAR_bayes.az.to_netcdf'), \
            patch('mimic.model_infer.infer_VAR_bayes.np.savez'):
        model.run_inference_large(samples=1000, tune=500, cores=4)
        mock_sample.assert_called_once_with(draws=1000, tune=500, cores=4)


def test_run_inference_xs(example_data, example_metabolite_data):
    # Initialize the model with data and metabolite data
    model = infer_VAR(data=example_data, dataS=example_metabolite_data)

    nX = example_data.shape[1]
    nS = example_metabolite_data.shape[1]

    # Create mock trace data for Ah and Bh
    mock_trace = create_mock_trace_xs(nX=nX, nS=nS)

    # Mock pymc's sample method
    with patch.object(pm, 'sample', return_value=mock_trace) as mock_sample, \
            patch('mimic.model_infer.infer_VAR_bayes.plt.savefig'), \
            patch('mimic.model_infer.infer_VAR_bayes.az.to_netcdf'), \
            patch('mimic.model_infer.infer_VAR_bayes.np.savez'):
        model.run_inference_xs(samples=500, tune=200, cores=2)


def test_posterior_analysis(mocker, example_data, example_metabolite_data):
    nX = example_data.shape[1]
    nS = example_metabolite_data.shape[1]
    mock_inference_data = create_mock_trace_xs(nX=nX, nS=nS)

    # Mock arviz.from_netcdf to return the mock inference data
    mocker.patch('arviz.from_netcdf', return_value=mock_inference_data)

    model = infer_VAR(data=example_data, dataS=example_metabolite_data)

    # Mock plotting and saving methods
    mocker.patch('mimic.model_infer.infer_VAR_bayes.plt.savefig')
    mocker.patch('mimic.model_infer.infer_VAR_bayes.az.to_netcdf')
    mocker.patch('mimic.model_infer.infer_VAR_bayes.np.savez')
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

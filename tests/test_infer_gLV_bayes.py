import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from mimic.model_infer.infer_gLV_bayes import infergLVbayes
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


@pytest.fixture
def mock_data():
    X = np.random.rand(10, 5)
    F = np.random.rand(10, 5)
    mu = np.random.rand(1, 5)
    M = np.random.rand(5, 5)
    return X, F, mu, M


@pytest.fixture
def bayes_model(mock_data):
    X, F, mu, M = mock_data
    return infergLVbayes(X=X, F=F, mu=mu, M=M, DA=10, DA0=1, N=10, noise_stddev=0.1, epsilon=np.random.rand(1, 5))


def test_import_data(mocker):
    mocker.patch('numpy.genfromtxt', return_value=np.array([[1, 2], [3, 4]]))
    model = infergLVbayes()
    model.import_data('fake_path.csv')
    assert model.data.shape == (2, 2)
    np.testing.assert_array_equal(model.data, np.array([[1, 2], [3, 4]]))


def generate_mock_inference_data():
    # Create mock data
    chains = 2
    draws = 100

    # Mock data for each variable
    mu_hat_data = np.random.rand(chains, draws, 5)
    M_ii_hat_data = np.random.rand(chains, draws, 5)
    M_ij_hat_data = np.random.rand(chains, draws, 5, 4)
    M_hat_data = np.random.rand(chains, draws, 5, 5)
    epsilon_hat_data = np.random.rand(chains, draws, 5)

    # Creating DataArray objects
    mu_hat = xr.DataArray(mu_hat_data, dims=("chain", "draw", "mu_hat_dim"))
    M_ii_hat = xr.DataArray(M_ii_hat_data, dims=("chain", "draw", "mu_hat_dim"))
    M_ij_hat = xr.DataArray(M_ij_hat_data, dims=("chain", "draw", "M_ij_dim_1", "M_ij_dim_2"))
    M_hat = xr.DataArray(M_hat_data, dims=("chain", "draw", "M_hat_dim_1", "M_hat_dim_2"))
    epsilon_hat = xr.DataArray(epsilon_hat_data, dims=("chain", "draw", "mu_hat_dim"))

    # Convert to an xarray.Dataset
    posterior_dataset = xr.Dataset({
        "mu_hat": mu_hat,
        "M_ii_hat": M_ii_hat,
        "M_ij_hat": M_ij_hat,
        "M_hat": M_hat,
        "epsilon_hat": epsilon_hat,  # Include epsilon_hat in the dataset
    })

    # Constructing the InferenceData object
    return az.InferenceData(posterior=posterior_dataset)


@patch('mimic.model_infer.infer_gLV_bayes.az.plot_posterior')
@patch('mimic.model_infer.infer_gLV_bayes.plt.savefig')
@patch('mimic.model_infer.infer_gLV_bayes.plt.show')
@patch('mimic.model_infer.infer_gLV_bayes.plt.close')
@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV(mock_sample, mock_close, mock_show, mock_savefig, mock_plot_posterior, bayes_model):
    bayes_model.run_bayes_gLV()
    mock_sample.assert_called_once()


@patch('mimic.model_infer.infer_gLV_bayes.az.plot_posterior')
@patch('mimic.model_infer.infer_gLV_bayes.plt.savefig')
@patch('mimic.model_infer.infer_gLV_bayes.plt.show')
@patch('mimic.model_infer.infer_gLV_bayes.plt.close')
@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV_shrinkage(mock_sample, mock_close, mock_show, mock_savefig, mock_plot_posterior, bayes_model):
    bayes_model.run_bayes_gLV_shrinkage()
    mock_sample.assert_called_once()


@patch('mimic.model_infer.infer_gLV_bayes.az.plot_posterior')
@patch('mimic.model_infer.infer_gLV_bayes.plt.savefig')
@patch('mimic.model_infer.infer_gLV_bayes.plt.show')
@patch('mimic.model_infer.infer_gLV_bayes.plt.close')
@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV_shrinkage_pert(mock_sample, mock_close, mock_show, mock_savefig, mock_plot_posterior, bayes_model):
    bayes_model.run_bayes_gLV_shrinkage_pert()
    mock_sample.assert_called_once()


@patch('mimic.model_infer.infer_gLV_bayes.az.plot_posterior')
@patch('mimic.model_infer.infer_gLV_bayes.plt.savefig')
@patch('mimic.model_infer.infer_gLV_bayes.plt.show')
@patch('mimic.model_infer.infer_gLV_bayes.plt.close')
def test_plot_posterior(mock_close, mock_show, mock_savefig, mock_plot_posterior, bayes_model):
    idata = generate_mock_inference_data()
    bayes_model.plot_posterior(idata, bayes_model.mu, bayes_model.M)
    assert mock_plot_posterior.call_count == 3
    assert mock_savefig.call_count == 3


@patch('mimic.model_infer.infer_gLV_bayes.az.plot_posterior')
@patch('mimic.model_infer.infer_gLV_bayes.plt.savefig')
@patch('mimic.model_infer.infer_gLV_bayes.plt.show')
@patch('mimic.model_infer.infer_gLV_bayes.plt.close')
def test_plot_posterior_pert(mock_close, mock_show, mock_savefig, mock_plot_posterior, bayes_model):
    idata = generate_mock_inference_data()
    bayes_model.plot_posterior_pert(idata, bayes_model.mu, bayes_model.M, bayes_model.epsilon)
    assert mock_plot_posterior.call_count == 4
    assert mock_savefig.call_count == 4


@patch('mimic.model_infer.infer_gLV_bayes.sns.heatmap')
@patch('mimic.model_infer.infer_gLV_bayes.plt.subplots')
def test_plot_interaction_matrix(mock_subplots, mock_heatmap, bayes_model):
    # Set up the return value for plt.subplots
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, ax_mock)
    M = np.random.rand(5, 5)
    M_h = np.random.rand(5, 5)
    bayes_model.plot_interaction_matrix(M, M_h)
    mock_heatmap.assert_called_once_with(M_h, ax=ax_mock, cmap='viridis')

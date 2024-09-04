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
    M_ii_hat = xr.DataArray(M_ii_hat_data, dims=(
        "chain", "draw", "mu_hat_dim"))
    M_ij_hat = xr.DataArray(M_ij_hat_data, dims=(
        "chain", "draw", "M_ij_dim_1", "M_ij_dim_2"))
    M_hat = xr.DataArray(M_hat_data, dims=(
        "chain", "draw", "M_hat_dim_1", "M_hat_dim_2"))
    epsilon_hat = xr.DataArray(
        epsilon_hat_data, dims=("chain", "draw", "mu_hat_dim"))

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


@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV(mock_sample, bayes_model):
    bayes_model.run_bayes_gLV()
    mock_sample.assert_called_once()


@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV_shrinkage(mock_sample, bayes_model):
    bayes_model.run_bayes_gLV_shrinkage()
    mock_sample.assert_called_once()


@patch('pymc.sample', return_value=generate_mock_inference_data())
def test_run_bayes_gLV_shrinkage_pert(mock_sample, bayes_model):
    bayes_model.run_bayes_gLV_shrinkage_pert()
    mock_sample.assert_called_once()


@patch('arviz.plot_posterior')
@patch('matplotlib.pyplot.savefig')
def test_plot_posterior(mock_savefig, mock_plot_posterior, bayes_model):
    idata = generate_mock_inference_data()
    bayes_model.plot_posterior(idata, bayes_model.mu, bayes_model.M)
    assert mock_plot_posterior.call_count == 3
    assert mock_savefig.call_count == 3


@patch('arviz.plot_posterior')
@patch('matplotlib.pyplot.savefig')
def test_plot_posterior_pert(mock_savefig, mock_plot_posterior, bayes_model):
    idata = generate_mock_inference_data()
    bayes_model.plot_posterior_pert(
        idata, bayes_model.mu, bayes_model.M, bayes_model.epsilon)
    assert mock_plot_posterior.call_count == 4
    assert mock_savefig.call_count == 4


@patch('seaborn.heatmap')
@patch('matplotlib.pyplot.subplots', return_value=(None, MagicMock()))
@patch('matplotlib.pyplot.savefig')
def plot_interaction_matrix(self, M, M_h):
    # visualize the interaction matrix
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Heatmap for M_hat
    sns.heatmap(M_h, ax=ax, cmap='viridis')
    ax.set_title('M_hat')
    ax.set_ylabel('X')
    ax.set_xlabel('X')

    # Annotate the true values for matrix1
    for i in range(M_h.shape[0]):
        for j in range(M_h.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                f'{M[i, j]:.2f}',
                ha='center',
                va='center',
                color='white')

    # Ensure that the figure is saved
    plt.savefig("interaction_matrix.pdf")
    plt.close(fig)

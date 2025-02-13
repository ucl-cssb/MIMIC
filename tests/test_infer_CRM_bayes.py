import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from mimic.model_infer.infer_CRM_bayes import inferCRMbayes
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import pytensor.tensor as tt


@pytest.fixture
def setup_data(request):
    """
    Fixture to provide mock data for testing.
    """
    # Generate mock data for testing purposes with 2 species
    num_species = 2
    num_resources = 2
    X = np.random.randn(100, num_species+1)  # Random design matrix for 100 samples and 2 species
    F = np.random.randn(100, num_species)  # Random data matrix for 100 samples and 2 species

    
    prior_tau_mean = 0
    prior_tau_sigma = 1
    prior_m_mean = -0.1
    prior_m_sigma = 0.1
    prior_r_mean = 0
    prior_r_sigma = 1
    prior_w_mean = -0.1
    prior_w_sigma = 0.1
    prior_K_mean = 0
    prior_K_sigma = 1
    prior_c_mean = -0.1
    prior_c_sigma = 0.1

    noise_stddev = 0.1
    draws = 500
    tune = 500
    chains = 4
    cores = 4
    
    return {
        "X": X,
        "F": F,
        "prior_tau_mean": prior_tau_mean,
        "prior_tau_sigma": prior_tau_sigma,
        "prior_m_mean": prior_m_mean,
        "prior_m_sigma": prior_m_sigma,
        "prior_r_mean": prior_r_mean,
        "prior_r_sigma": prior_r_sigma,
        "prior_w_mean": prior_w_mean,
        "prior_w_sigma": prior_w_sigma,
        "prior_K_mean": prior_K_mean,
        "prior_K_sigma": prior_K_sigma,
        "prior_c_mean": prior_c_mean,
        "prior_c_sigma": prior_c_sigma,
        "noise_stddev": noise_stddev,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores
    }


@pytest.fixture
def bayes_CRM_instance(setup_data):
    """
    Fixture to create an instance of the infergLVbayes with mock data.
    """
    data = setup_data
    # Initialize the object without any parameters
    bayes_CRM = inferCRMbayes()

    # Set the parameters using the set_parameters method
    bayes_CRM.set_parameters(
        X=data["X"], 
        F=data["F"],
        prior_tau_mean=data["prior_tau_mean"],
        prior_tau_sigma=data["prior_tau_sigma"],
        prior_m_mean=data["prior_m_mean"],
        prior_m_sigma=data["prior_m_sigma"],
        prior_r_mean=data["prior_r_mean"],
        prior_r_sigma=data["prior_r_sigma"],
        prior_w_mean=data["prior_w_mean"],
        prior_w_sigma=data["prior_w_sigma"],
        prior_K_mean=data["prior_K_mean"],
        prior_K_sigma=data["prior_K_sigma"],
        prior_c_mean=data["prior_c_mean"],
        prior_c_sigma=data["prior_c_sigma"],
        noise_stddev=data["noise_stddev"],
        draws=data["draws"],
        tune=data["tune"],
        chains=data["chains"],
        cores=data["cores"]
    )

    return bayes_CRM


def test_run_inference(bayes_CRM_instance):
    """
    Test the `_bayes_run_inference` function to check if it returns the correct output without shrinkage or perturbation.
    """
    
    # Call the method to test without shrinkage or perturbation
    idata = bayes_CRM_instance.run_inference()

    # Check that the output is an ArviZ InferenceData object
    assert isinstance(idata, az.InferenceData), "The output is not an InferenceData object."

    # Check that the expected variables are in the posterior
    assert "tau_hat" in idata.posterior, "'tau_hat' is not in the posterior."
    assert "m_hat" in idata.posterior, "'m_hat' is not in the posterior."
    assert "r_hat" in idata.posterior, "'r_hat' is not in the posterior."
    assert "w_hat" in idata.posterior, "'w_hat' is not in the posterior."
    assert "K_hat" in idata.posterior, "'K_hat' is not in the posterior."
    assert "c_hat" in idata.posterior, "'c_hat' is not in the posterior."


    # Additional assertions to test the integrity of the returned posterior
    assert len(idata.posterior["tau_hat"]) > 0, "'tau_hat' has no samples."
    assert len(idata.posterior["m_hat"]) > 0, "'m_hat' has no samples."
    assert len(idata.posterior["r_hat"]) > 0, "'r_hat' has no samples."
    assert len(idata.posterior["w_hat"]) > 0, "'w_hat' has no samples."
    assert len(idata.posterior["K_hat"]) > 0, "'K_hat' has no samples."
    assert len(idata.posterior["c_hat"]) > 0, "'c_hat' has no samples."


def test_plot_posterior(bayes_CRM_instance):
    """
    Test the `plot_posterior` function to ensure it runs without errors.
    """
    # Generate the InferenceData from the method
    idata = bayes_CRM_instance.run_inference()

    # Try to call the plot_posterior method
    try:
        bayes_CRM_instance.plot_posterior(idata)
    except Exception as e:
        pytest.fail(f"plot_posterior raised an exception: {e}")
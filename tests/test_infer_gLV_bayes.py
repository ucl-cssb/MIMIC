import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from mimic.model_infer.infer_gLV_bayes import infergLVbayes
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
    X = np.random.randn(100, num_species+1)  # Random design matrix for 100 samples and 2 species
    F = np.random.randn(100, num_species)  # Random data matrix for 100 samples and 2 species

    # Add a condition for different shapes depending on number of parameters
    test_name = request.node.nodeid.split("::")[-1] 
    if test_name in ["test_run_inference", "test_run_inference_shrinkage"]:
        X = np.random.randn(100, num_species + 1)
    elif test_name in ["test_run_inference_shrinkage_pert", "test_plot_posterior_pert"]:
        X = np.random.randn(100, num_species + 2)

    
    prior_mu_mean = 0
    prior_mu_sigma = 1
    prior_Mii_mean = -0.1
    prior_Mii_sigma = 0.1
    prior_Mij_sigma = 0.1
    prior_eps_mean = 0
    prior_eps_sigma = 0.1
    DA = 0.1
    DA0 = 0.01
    N = 100
    noise_stddev = 0.1
    draws = 500
    tune = 1000
    chains = 2
    cores = 1
    
    return {
        "X": X,
        "F": F,
        "prior_mu_mean": prior_mu_mean,
        "prior_mu_sigma": prior_mu_sigma,
        "prior_Mii_mean": prior_Mii_mean,
        "prior_Mii_sigma": prior_Mii_sigma,
        "prior_Mij_sigma": prior_Mij_sigma,
        "prior_eps_mean": prior_eps_mean,
        "prior_eps_sigma": prior_eps_sigma,
        "DA": DA,
        "DA0": DA0,
        "N": N,
        "noise_stddev": noise_stddev,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores
    }


@pytest.fixture
def bayes_gLV_instance(setup_data):
    """
    Fixture to create an instance of the infergLVbayes with mock data.
    """
    data = setup_data
    # Initialize the object without any parameters
    bayes_gLV = infergLVbayes()

    # Set the parameters using the set_parameters method
    bayes_gLV.set_parameters(
        X=data["X"], 
        F=data["F"],
        prior_mu_mean=data["prior_mu_mean"],
        prior_mu_sigma=data["prior_mu_sigma"],
        prior_Mii_mean=data["prior_Mii_mean"],
        prior_Mii_sigma=data["prior_Mii_sigma"],
        prior_Mij_sigma=data["prior_Mij_sigma"],
        prior_eps_mean=data["prior_eps_mean"],
        prior_eps_sigma=data["prior_eps_sigma"],
        DA=data["DA"],
        DA0=data["DA0"],
        N=data["N"],
        noise_stddev=data["noise_stddev"],
        draws=data["draws"],
        tune=data["tune"],
        chains=data["chains"],
        cores=data["cores"]
    )

    return bayes_gLV


def test_run_inference(bayes_gLV_instance):
    """
    Test the `_bayes_run_inference` function to check if it returns the correct output without shrinkage or perturbation.
    """
    
    # Call the method to test without shrinkage or perturbation
    idata = bayes_gLV_instance.run_inference()

    # Check that the output is an ArviZ InferenceData object
    assert isinstance(idata, az.InferenceData), "The output is not an InferenceData object."

    # Check that the expected variables are in the posterior
    assert "mu_hat" in idata.posterior, "'mu_hat' is not in the posterior."
    assert "M_hat" in idata.posterior, "'M_hat' is not in the posterior."

    # Additional assertions to test the integrity of the returned posterior
    assert len(idata.posterior["mu_hat"]) > 0, "'mu_hat' has no samples."
    assert len(idata.posterior["M_hat"]) > 0, "'M_hat' has no samples."


def test_run_inference_shrinkage(bayes_gLV_instance):
    """
    Test the `run_bayes_gLV_shrinkage` function to check if it returns the correct output.
    """
    # Call the method to test
    idata = bayes_gLV_instance.run_inference_shrinkage()

    # Check that the output is an ArviZ InferenceData object
    assert isinstance(idata, az.InferenceData), "The output is not an InferenceData object."

    # Check that the expected variables are in the posterior
    assert "mu_hat" in idata.posterior, "'mu_hat' is not in the posterior."
    assert "M_hat" in idata.posterior, "'M_hat' is not in the posterior."


def test_run_inference_shrinkage_pert(bayes_gLV_instance):
    """
    Test the `run_bayes_gLV_shrinkage_pert` function to check if it returns the correct output.
    """
    # Call the method to test
    idata = bayes_gLV_instance.run_inference_shrinkage_pert()

    # Check that the output is an ArviZ InferenceData object
    assert isinstance(idata, az.InferenceData), "The output is not an InferenceData object."

    # Check that the expected variables are in the posterior
    assert "mu_hat" in idata.posterior, "'mu_hat' is not in the posterior."
    assert "M_hat" in idata.posterior, "'M_hat' is not in the posterior."
    assert "epsilon_hat" in idata.posterior, "'epsilon_hat' is not in the posterior."


def test_plot_posterior(bayes_gLV_instance):
    """
    Test the `plot_posterior` function to ensure it runs without errors.
    """
    # Generate the InferenceData from the method
    idata = bayes_gLV_instance.run_inference_shrinkage()

    # Try to call the plot_posterior method
    try:
        bayes_gLV_instance.plot_posterior(idata)
    except Exception as e:
        pytest.fail(f"plot_posterior raised an exception: {e}")


def test_plot_posterior_pert(bayes_gLV_instance):
    """
    Test the `plot_posterior_pert` function to ensure it produces the correct plot.
    """
    # First, generate the InferenceData from the method
    idata = bayes_gLV_instance.run_inference_shrinkage_pert()

    # Try to call the plot_posterior_pert method
    try:
        bayes_gLV_instance.plot_posterior_pert(idata)
    except Exception as e:
        pytest.fail(f"plot_posterior_pert raised an exception: {e}")
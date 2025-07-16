import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from mimic.model_infer.infer_CRM_bayes import inferCRMbayes
from mimic.model_simulate.sim_CRM import sim_CRM
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
    times = np.arange(0, 10, 0.1)


    tau = np.random.uniform(0.1, 0.9, num_species)      # species timescales
    w = np.random.uniform(0.1, 0.9, num_resources)      # resource quality
    c = np.random.uniform(0.1, 0.9, (num_species, num_resources))   # relative resource preferences
    m = np.random.uniform(0.3, 0.7, num_species)        # mortality rates
    r = np.random.uniform(0.1, 0.9, num_resources)      # resource timescales
    K = np.random.uniform(1.0, 10.0, num_resources)     # resource carrying capacities

    # initial conditions
    init_species = 10 * np.ones(num_species+num_resources)

    # instantiate simulator

    simulator = sim_CRM()

    simulator.set_parameters(num_species = num_species,
                            num_resources = num_resources,
                            tau = tau,
                            w = w,
                            c = c,
                            m = m,
                            r = r,
                            K = K)


    observed_species, observed_resources = simulator.simulate(times, init_species)

    # add Guassian noise to the data
    observed_species = observed_species + np.random.normal(loc=0, scale=0.1, size=observed_species.shape)
    observed_resources = observed_resources + np.random.normal(loc=0, scale=0.1, size=observed_resources.shape)

    # Replace negative values with 0
    observed_species = np.maximum(observed_species, 0.0001)
    observed_resources = np.maximum(observed_resources, 0.0001)

    yobs = np.hstack((observed_species, observed_resources))


    # num_species = 2
    # num_resources = 2
    # times = np.random.randn(100, num_species+1)  # 
    # yobs = np.random.randn(100, num_species)  # 

    # tau = np.array([0.7, 0.3])
    # m = np.array([0.5, 0.4])
    # r = np.array([0.4, 0.7])
    # w = np.array([0.3, 0.5])
    # K = np.array([5.0, 1.5])


    prior_c_mean = 0.6
    prior_c_sigma = 0.2
    num_species = 2
    num_resources = 2

    draws = 20
    tune = 20
    chains = 4
    cores = 4
    
    return {
        "times": times,
        "yobs": yobs,
        "tau": tau,
        "m": m,
        "r": r,
        "w": w,
        "K": K,
        "prior_c_mean": prior_c_mean,
        "prior_c_sigma": prior_c_sigma,
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
        times=data["times"], 
        yobs=data["yobs"],
        num_species=2,
        num_resources=2,
        tau=data["tau"],
        m=data["m"],
        r=data["r"],
        w=data["w"],
        K=data["K"],
        prior_c_mean=data["prior_c_mean"],
        prior_c_sigma=data["prior_c_sigma"],
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
    # assert "tau_hat" in idata.posterior, "'tau_hat' is not in the posterior."
    # assert "m_hat" in idata.posterior, "'m_hat' is not in the posterior."
    # assert "r_hat" in idata.posterior, "'r_hat' is not in the posterior."
    # assert "w_hat" in idata.posterior, "'w_hat' is not in the posterior."
    # assert "K_hat" in idata.posterior, "'K_hat' is not in the posterior."
    assert "c_hat" in idata.posterior, "'c_hat' is not in the posterior."


    # Additional assertions to test the integrity of the returned posterior
    # assert len(idata.posterior["tau_hat"]) > 0, "'tau_hat' has no samples."
    # assert len(idata.posterior["m_hat"]) > 0, "'m_hat' has no samples."
    # assert len(idata.posterior["r_hat"]) > 0, "'r_hat' has no samples."
    # assert len(idata.posterior["w_hat"]) > 0, "'w_hat' has no samples."
    # assert len(idata.posterior["K_hat"]) > 0, "'K_hat' has no samples."
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

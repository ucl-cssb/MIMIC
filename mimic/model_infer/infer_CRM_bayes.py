import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as at
import pickle
import cloudpickle

from mimic.utilities import *
from mimic.model_simulate.sim_gLV import *

from mimic.model_infer.base_infer import BaseInfer

import os
from typing import Optional, Union, List, Dict, Any


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_growth_curves(data, ax=None):
    # Will show plot automatically if singular, or handle faceting if ax is given
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting X1_bar
    ax.plot(data['time'], data['X1_bar'], label='X1_bar')

    # Check if 'X2_bar' exists and plot it
    if 'X2_bar' in data.columns:
        ax.plot(data['time'], data['X2_bar'], label='X2_bar')

    # Adding labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Growth Curves of X1_bar (and X2_bar if available)')

    # Adding a legend
    #ax.legend()

    # If ax was not provided, show the plot
    if ax is None:
        plt.show()




class inferCRMbayes(BaseInfer):
    """
    inferCRMbayes class for Bayesian inference of CRM models.

    Args:
        X (np.ndarray): The design matrix
        F (np.ndarray): The observed values

    Methods:
        set_parameters: Set or update simulation parameters.
        run_inference: Run Bayesian inference with uniform priors and specified bounds.

    Returns:
        None
    """

    def __init__(
            self,
            num_species=2,
            num_resources=0): 

        super().__init__()  # Call base class constructor

        self.X: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None

        self.num_species = num_species
        self.num_resources = num_resources

        # Default values
        self.tau = numpy.ones(num_species),  # species_timescale
        self.r = numpy.ones(num_resources),  # resource_timescale

        self.w = numpy.ones(num_resources),  # resource_quality
        self.c = numpy.ones((num_species, num_resources)),  # relative_preference
        self.m = numpy.ones(num_species),  # mortality_rate

        self.K = numpy.ones(num_resources)  # resource_capacity

        # Sampling parameters
        self.draws: Optional[int] = None
        self.tune: Optional[int] = None
        self.chains: Optional[int] = None
        self.cores: Optional[int] = None

        # Parameter bounds
        self.tau_bounds = (0, 2)
        self.r_bounds = (0, 2)
        self.w_bounds = (0, 2)
        self.c_bounds = (0, 2)
        self.m_bounds = (0, 1)
        self.K_bounds = (0, 2)

        self.parameters = {
            "num_species": self.num_species,
            "num_resources": self.num_resources,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K}

    def set_parameters(self,
                       X: Optional[np.ndarray] = None,
                       F: Optional[np.ndarray] = None,
                       num_species: Optional[int] = None,
                       num_resources: Optional[int] = None,
                       tau: Optional[Union[List[float], numpy.ndarray]] = None,
                       w: Optional[Union[List[float], numpy.ndarray]] = None,
                       c: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       m: Optional[Union[List[float], numpy.ndarray]] = None,
                       r: Optional[Union[List[float], numpy.ndarray]] = None,
                       K: Optional[Union[List[float], numpy.ndarray]] = None,
                       tau_bounds: Optional[tuple] = None,
                       r_bounds: Optional[tuple] = None,
                       w_bounds: Optional[tuple] = None,
                       c_bounds: Optional[tuple] = None,
                       m_bounds: Optional[tuple] = None,
                       K_bounds: Optional[tuple] = None,
                       draws: Optional[int] = None,
                       tune: Optional[int] = None,
                       chains: Optional[int] = None,
                       cores: Optional[int] = None) -> None:

        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            num_resources (Optional[int]): Number of resources.
            tau (Optional[Union[List[float], numpy.ndarray]]): species_timescales.
            w (Optional[Union[List[float], numpy.ndarray]]): resource_quality.
            c (Optional[Union[List[List[float]], numpy.ndarray]]): relative_preference.
            m (Optional[Union[List[float], numpy.ndarray]]): mortality_rate.
            r (Optional[Union[List[float], numpy.ndarray]]): resource timescale.
            K (Optional[Union[List[float], numpy.ndarray]]): resource capacity.
            tau_bounds, r_bounds, w_bounds, c_bounds, m_bounds, K_bounds (Optional[tuple]): Lower and upper bounds for uniform priors.
        """

        if X is not None:
            self.X = np.array(X)
        if F is not None:
            self.F = np.array(F)
        if num_species is not None:
            self.num_species = num_species
        if num_resources is not None:
            self.num_resources = num_resources
        if tau is not None:
            self.tau = tau
        if w is not None:
            self.w = w
        if c is not None:
            self.c = numpy.asarray(c, dtype=numpy.float64)
        if m is not None:
            self.m = m
        if r is not None:
            self.r = r
        if K is not None:
            self.K = K

        # Update bounds if provided
        if tau_bounds is not None:
            self.tau_bounds = tau_bounds
        if r_bounds is not None:
            self.r_bounds = r_bounds
        if w_bounds is not None:
            self.w_bounds = w_bounds
        if c_bounds is not None:
            self.c_bounds = c_bounds
        if m_bounds is not None:
            self.m_bounds = m_bounds
        if K_bounds is not None:
            self.K_bounds = K_bounds

        if draws is not None:
            self.draws = draws
        if tune is not None:
            self.tune = tune
        if chains is not None:
            self.chains = chains
        if cores is not None:
            self.cores = cores

        self.parameters = {
            "X": self.X,
            "F": self.F,
            "num_species": self.num_species,
            "num_resources": self.num_resources,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K,
            "tau_bounds": self.tau_bounds,
            "w_bounds": self.w_bounds,
            "c_bounds": self.c_bounds,
            "m_bounds": self.m_bounds,
            "r_bounds": self.r_bounds,
            "K_bounds": self.K_bounds,
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores}


    def run_inference(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model

        Returns:
            idata: The posterior inference data
            var_names: Names of each M(ij) parameter used


        """

        if self.X is None or self.F is None:
            raise ValueError("X, F must both be provided.")

        # data = self.data
        X = self.X
        F = self.F
        num_species = self.num_species
        num_resources = self.num_resources
        tau = self.tau
        w = self.w
        c = self.c
        m = self.m
        r = self.r
        K = self.K
        tau_bounds = self.tau_bounds
        r_bounds = self.r_bounds
        w_bounds = self.w_bounds
        c_bounds = self.c_bounds
        m_bounds = self.m_bounds
        K_bounds = self.K_bounds
        draws = self.draws
        tune = self.tune
        chains = self.chains
        cores = self.cores


        # Print shapes to ensure data is correct
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")
        print(f"Number of resources: {num_resources}")

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters

            # species timescales
            tau_hat = pm.Uniform('tau', lower=tau_bounds[0], upper=tau_bounds[1], shape=(num_species,))

            # resource timescales
            r_hat = pm.Uniform('r', lower=r_bounds[0], upper=r_bounds[1], shape=(num_resources,))

            # resource quality
            w_hat = pm.Uniform('w', lower=w_bounds[0], upper=w_bounds[1], shape=(num_resources,))

            # relative resource preferences
            c_hat_vals = pm.Uniform('c_hat_vals', lower=c_bounds[0], upper=c_bounds[1], shape=(num_species, num_resources))

            c_hat = pm.Deterministic('c_hat', c_hat_vals)

            # mortality rates
            m_hat = pm.Uniform('m', lower=m_bounds[0], upper=m_bounds[1], shape=(num_species,))

            # resource carrying capacities
            K_hat = pm.Uniform('K', lower=K_bounds[0], upper=K_bounds[1], shape=(num_resources,))

            print("Shape of tau_hat:", tau_hat.shape.eval())
            print("Shape of c_hat:", c_hat.shape.eval())

            # Expected value of outcome
            combined_values = pm.math.concatenate([c_hat, tau_hat], axis=0)
            print("Combined values shape:", combined_values.shape.eval())
            model_mean = pm.math.dot(X, combined_values)

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=1, observed=F)  # Assume sigma=1, update if needed

            # Posterior distribution sampling
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, progressbar=True)

        return idata

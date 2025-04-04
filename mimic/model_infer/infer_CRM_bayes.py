import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pytensor.tensor as at
import pytensor
import pickle
import cloudpickle
import os
from typing import Optional, Union, List, Dict, Any

import pymc as pm
from pymc.ode import DifferentialEquation

from scipy import stats
from scipy.integrate import odeint

from mimic.utilities import *
from mimic.model_simulate.sim_CRM import *
from mimic.model_infer.base_infer import BaseInfer


def plot_growth_curves(data, ax=None):
    # Will show plot automatically if singular, or handle faceting if ax is
    # given
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
    # ax.legend()

    # If ax was not provided, show the plot
    if ax is None:
        plt.show()

# Define the ODE function for CRM model


def CRM_inf_func(y, t, p):
    # Unpack parameters from the vector p
    nr = p[0].astype("int32")   # Number of resources
    nsp = p[1].astype("int32")   # Number of species
    tau = p[2:2 + nsp]  # Species time scales
    w = p[2 + nsp:2 + nsp + nr]  # Resource quality
    # Flattened resource preferences
    c_flat = p[2 + nsp + nr:2 + nsp + nr + (nsp * nr)]
    c = c_flat.reshape((nsp, nr))  # Reshape to nsp x nr matrix
    m = p[2 + nsp + nr + (nsp * nr):2 + (2 * nsp) + nr +
          (nsp * nr)]  # Mortality rates
    r = p[2 + (2 * nsp) + nr + (nsp * nr):2 + (2 * nsp) +
          (2 * nr) + (nsp * nr)]  # Resource time scales
    K = p[2 + (2 * nsp) + (2 * nr) + (nsp * nr):2 + (2 * nsp) +
          (3 * nr) + (nsp * nr)]  # Resource carrying capacities

    # Separate species (N) and resources (R)
    N = y[:nsp]  # Species populations
    R = y[nsp:]  # Resource availability

    # Species growth equation (dN)
    growth_term = at.dot(c, w * R)  # Matrix multiplication as tensor
    dN = (N / tau) * (growth_term - m)  # Species growth equation

    # Resource consumption equation (dR)
    consumption_term = at.dot(N, c)  # Matrix multiplication as tensor
    dR = (1 / (r * K)) * (K - R) * R - consumption_term * \
        R  # Resource consumption equation

    # Combine dN and dR into a single 1D array
    derivatives = [dN[0], dN[1], dR[0], dR[1]]  # 1D array
    # derivatives = np.concatenate([dN, dR])  # Concatenate species and
    # resource derivatives

    # Return the derivatives for both species and resources as a single array
    return derivatives


class inferCRMbayes(BaseInfer):
    """
    inferCRMbayes class for Bayesian inference of CRM models.

    Args:
        times (np.ndarray): The times at which the observations were made
        yobs (np.ndarray): The observed values
        num_species (int): The number of species
        num_resources (int): The number of resources
        tau (int, float, List)): The species timescales
        w (int, float, List)): The resource quality
        c (int, float, List)): The relative preference
        m (int, float, List)): The mortality rate
        r (int, float, List)): The resource timescale
        K (int, float, List)): The resource capacity

        prior_tau_mean (int, float, List): The mean of the prior for species timescales
        prior_tau_sigma (int, float, List): The standard deviation of the prior for species timescales
        prior_w_mean (int, float, List): The mean of the prior for resource quality
        prior_w_sigma (int, float, List): The standard deviation of the prior for resource quality
        prior_c_mean (int, float, List): The mean of the prior for relative preference
        prior_c_sigma (int, float, List): The standard deviation of the prior for relative preference
        prior_m_mean (int, float, List): The mean of the prior for mortality rate
        prior_m_sigma (int, float, List): The standard deviation of the prior for mortality rate
        prior_r_mean (int, float, List): The mean of the prior for resource timescale
        prior_r_sigma (int, float, List): The standard deviation of the prior for resource timescale
        prior_K_mean (int, float, List): The mean of the prior for resource capacity
        prior_K_sigma (int, float, List): The standard deviation of the prior for resource capacity


    Methods:
        set_parameters: Set or update simulation parameters.
        run_inference: Run Bayesian inference with uniform priors and specified bounds.
        plot_posterior: Plot the posterior distributions of the inferred parameters.

    Returns:
        None
    """

    def __init__(
            self,
            times=None,
            yobs=None,
            num_species=None,
            num_resources=None,
            prior_tau_mean=None,
            prior_tau_sigma=None,
            prior_w_mean=None,
            prior_w_sigma=None,
            prior_c_mean=None,
            prior_c_sigma=None,
            prior_m_mean=None,
            prior_m_sigma=None,
            prior_r_mean=None,
            prior_r_sigma=None,
            prior_K_mean=None,
            prior_K_sigma=None):

        super().__init__()  # Call base class constructor

        self.times: Optional[np.ndarray] = times
        self.yobs: Optional[np.ndarray] = yobs

        self.num_species: Optional[int] = num_species
        self.num_resources: Optional[int] = num_resources

        # Default values
        self.tau: Optional[Union[int, float]] = None  # species_timescale
        self.r: Optional[Union[int, float]] = None  # resource_timescale
        self.w: Optional[Union[int, float]] = None  # resource_quality
        self.c: Optional[Union[int, float]] = None  # relative_preference
        self.m: Optional[Union[int, float]] = None  # mortality_rate
        self.K: Optional[Union[int, float]] = None  # resource_capacity

        # Sampling parameters
        self.draws: Optional[int] = None
        self.tune: Optional[int] = None
        self.chains: Optional[int] = None
        self.cores: Optional[int] = None

        # Priors
        self.prior_tau_mean: Optional[Union[int, float,
                                            List[Union[int, float]]]] = prior_tau_mean
        self.prior_tau_sigma: Optional[Union[int, float,
                                             List[Union[int, float]]]] = prior_tau_sigma
        self.prior_w_mean: Optional[Union[int, float,
                                          List[Union[int, float]]]] = prior_w_mean
        self.prior_w_sigma: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_w_sigma
        self.prior_c_mean: Optional[Union[int, float,
                                          List[Union[int, float]]]] = prior_c_mean
        self.prior_c_sigma: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_c_sigma
        self.prior_m_mean: Optional[Union[int, float,
                                          List[Union[int, float]]]] = prior_m_mean
        self.prior_m_sigma: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_m_sigma
        self.prior_r_mean: Optional[Union[int, float,
                                          List[Union[int, float]]]] = prior_r_mean
        self.prior_r_sigma: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_r_sigma
        self.prior_K_mean: Optional[Union[int, float,
                                          List[Union[int, float]]]] = prior_K_mean
        self.prior_K_sigma: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_K_sigma

        self.parameters: Dict[str,
                              Optional[Union[int,
                                             float,
                                             np.ndarray,
                                             str]]] = {"num_species": self.num_species,
                                                       "num_resources": self.num_resources,
                                                       "tau": self.tau,
                                                       "w": self.w,
                                                       "c": self.c,
                                                       "m": self.m,
                                                       "r": self.r,
                                                       "K": self.K,
                                                       "prior_tau_mean": self.prior_tau_mean,
                                                       "prior_tau_sigma": self.prior_tau_sigma,
                                                       "prior_w_mean": self.prior_w_mean,
                                                       "prior_w_sigma": self.prior_w_sigma,
                                                       "prior_c_mean": self.prior_c_mean,
                                                       "prior_c_sigma": self.prior_c_sigma,
                                                       "prior_m_mean": self.prior_m_mean,
                                                       "prior_m_sigma": self.prior_m_sigma,
                                                       "prior_r_mean": self.prior_r_mean,
                                                       "prior_r_sigma": self.prior_r_sigma,
                                                       "prior_K_mean": self.prior_K_mean,
                                                       "prior_K_sigma": self.prior_K_sigma}

    def set_parameters(self,
                       times: Optional[np.ndarray] = None,
                       yobs: Optional[np.ndarray] = None,
                       num_species: Optional[int] = None,
                       num_resources: Optional[int] = None,
                       tau: Optional[Union[List[float], np.ndarray]] = None,
                       w: Optional[Union[List[float], np.ndarray]] = None,
                       c: Optional[Union[List[List[float]], np.ndarray]] = None,
                       m: Optional[Union[List[float], np.ndarray]] = None,
                       r: Optional[Union[List[float], np.ndarray]] = None,
                       K: Optional[Union[List[float], np.ndarray]] = None,
                       prior_tau_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_tau_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_w_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_w_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_c_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_c_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_m_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_m_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_r_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_r_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_K_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_K_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       draws: Optional[int] = None,
                       tune: Optional[int] = None,
                       chains: Optional[int] = None,
                       cores: Optional[int] = None) -> None:
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            num_resources (Optional[int]): Number of resources.
            tau (Optional[Union[List[float], np.ndarray]]): species_timescales.
            w (Optional[Union[List[float], np.ndarray]]): resource_quality.
            c (Optional[Union[List[List[float]], np.ndarray]]): relative_preference.
            m (Optional[Union[List[float], np.ndarray]]): mortality_rate.
            r (Optional[Union[List[float], np.ndarray]]): resource timescale.
            K (Optional[Union[List[float], np.ndarray]]): resource capacity.
            prior_tau_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for species timescales.
            prior_tau_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for species timescales.
            prior_w_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for resource quality.
            prior_w_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for resource quality.
            prior_c_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for relative preference.
            prior_c_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for relative preference.
            prior_m_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for mortality rate.
            prior_m_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for mortality rate.
            prior_r_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for resource timescale.
            prior_r_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for resource timescale.
            prior_K_mean (Optional[Union[int, float, List[Union[int, float]]]]): The mean of the prior for resource capacity.
            prior_K_sigma (Optional[Union[int, float, List[Union[int, float]]]]): The standard deviation of the prior for resource capacity.
            draws (Optional[int]): Number of draws from the posterior distribution.
            tune (Optional[int]): Number of tuning steps.
            chains (Optional[int]): Number of chains.
            cores (Optional[int]): Number of cores.
        """

        if times is not None:
            self.times = np.array(times)
        if yobs is not None:
            self.yobs = np.array(yobs)
        if num_species is not None:
            self.num_species = num_species
        if num_resources is not None:
            self.num_resources = num_resources
        if tau is not None:
            self.tau = tau
        if w is not None:
            self.w = w
        if c is not None:
            self.c = np.asarray(c, dtype=np.float64)
        if m is not None:
            self.m = m
        if r is not None:
            self.r = r
        if K is not None:
            self.K = K

        # priors
        if prior_tau_mean is not None:
            self.prior_tau_mean = prior_tau_mean
        if prior_tau_sigma is not None:
            self.prior_tau_sigma = prior_tau_sigma
        if prior_w_mean is not None:
            self.prior_w_mean = prior_w_mean
        if prior_w_sigma is not None:
            self.prior_w_sigma = prior_w_sigma
        if prior_c_mean is not None:
            self.prior_c_mean = prior_c_mean
        if prior_c_sigma is not None:
            self.prior_c_sigma = prior_c_sigma
        if prior_m_mean is not None:
            self.prior_m_mean = prior_m_mean
        if prior_m_sigma is not None:
            self.prior_m_sigma = prior_m_sigma
        if prior_r_mean is not None:
            self.prior_r_mean = prior_r_mean
        if prior_r_sigma is not None:
            self.prior_r_sigma = prior_r_sigma
        if prior_K_mean is not None:
            self.prior_K_mean = prior_K_mean
        if prior_K_sigma is not None:
            self.prior_K_sigma = prior_K_sigma

        if draws is not None:
            self.draws = draws
        if tune is not None:
            self.tune = tune
        if chains is not None:
            self.chains = chains
        if cores is not None:
            self.cores = cores

        self.parameters = {
            "times": self.times,
            "yobs": self.yobs,
            "num_species": self.num_species,
            "num_resources": self.num_resources,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K,
            "prior_tau_mean": self.prior_tau_mean,
            "prior_tau_sigma": self.prior_tau_sigma,
            "prior_w_mean": self.prior_w_mean,
            "prior_w_sigma": self.prior_w_sigma,
            "prior_c_mean": self.prior_c_mean,
            "prior_c_sigma": self.prior_c_sigma,
            "prior_m_mean": self.prior_m_mean,
            "prior_m_sigma": self.prior_m_sigma,
            "prior_r_mean": self.prior_r_mean,
            "prior_r_sigma": self.prior_r_sigma,
            "prior_K_mean": self.prior_K_mean,
            "prior_K_sigma": self.prior_K_sigma,
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores}

    def import_data(self, file_path) -> None:
        """
        Imports data from a .csv file.

        Args:
        file_path (str): The path to the .csv file.

        Returns:
        None
        """
        self.data = np.genfromtxt(file_path, delimiter=',')
        return

    def run_inference(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model

        Returns:
            idata: The posterior inference data
            var_names: Names of each M(ij) parameter used


        """

        if self.times is None or self.yobs is None:
            raise ValueError("times, yobs must both be provided.")

        # data = self.data
        times = self.times
        yobs = self.yobs
        num_species = self.num_species
        num_resources = self.num_resources
        tau = self.tau
        w = self.w
        c = self.c
        m = self.m
        r = self.r
        K = self.K
        prior_tau_mean = self.prior_tau_mean
        prior_tau_sigma = self.prior_tau_sigma
        prior_w_mean = self.prior_w_mean
        prior_w_sigma = self.prior_w_sigma
        prior_c_mean = self.prior_c_mean
        prior_c_sigma = self.prior_c_sigma
        prior_m_mean = self.prior_m_mean
        prior_m_sigma = self.prior_m_sigma
        prior_r_mean = self.prior_r_mean
        prior_r_sigma = self.prior_r_sigma
        prior_K_mean = self.prior_K_mean
        prior_K_sigma = self.prior_K_sigma
        draws = self.draws
        tune = self.tune
        chains = self.chains
        cores = self.cores

        # Print shapes to ensure data is correct
        print(f"times shape: {times.shape}")
        print(f"yobs shape: {yobs.shape}")
        print(f"Number of species: {num_species}")
        print(f"Number of resources: {num_resources}")

        # Set up initial conditions and parameters
        nsp = num_species
        nr = num_resources
        n_states = nsp + nr
        n_theta = 2 + (2 * nsp) + (3 * nr) + (nsp * nr)

        # Define the DifferentialEquation model
        crm_model = DifferentialEquation(
            func=CRM_inf_func,  # The ODE function
            times=times,  # Time points for the solution
            n_states=n_states,  # Total state variables (nsp + nr)
            n_theta=n_theta,
            # Total number of parameters (2 + tau, w, c, m, r, K)
            t0=0  # Initial time
        )

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters

            sigma = pm.HalfNormal(
                'sigma', sigma=0.5, shape=(
                    1,))  # Same sigma for all responses

            # Conditionally define parameters based on whether priors are
            # provided

            # For tau parameter
            if prior_tau_mean is not None and prior_tau_sigma is not None:
                tau_hat = pm.TruncatedNormal(
                    'tau_hat',
                    mu=prior_tau_mean,
                    sigma=prior_tau_sigma,
                    lower=0.1,
                    shape=(
                        nsp,
                    ))
                print("tau_hat is inferred")
            else:
                tau_hat = at.as_tensor_variable(tau)
                print("tau_hat is fixed")

            # For w parameter
            if prior_w_mean is not None and prior_w_sigma is not None:
                w_hat = pm.TruncatedNormal(
                    'w_hat',
                    mu=prior_w_mean,
                    sigma=prior_w_sigma,
                    lower=0.1,
                    shape=(
                        nr,
                    ))
                print("w_hat is inferred")
            else:
                w_hat = at.as_tensor_variable(w)
                print("w_hat is fixed")

            # For c parameter
            if prior_c_mean is not None and prior_c_sigma is not None:
                c_hat_vals = pm.TruncatedNormal(
                    'c_hat_vals',
                    mu=prior_c_mean,
                    sigma=prior_c_sigma,
                    lower=0.1,
                    shape=(
                        nsp,
                        nr))
                c_hat = pm.Deterministic('c_hat', c_hat_vals)
                print("c_hat is inferred")
            else:
                c_hat = at.as_tensor_variable(c)
                print("c_hat is fixed")

            # For m parameter
            if prior_m_mean is not None and prior_m_sigma is not None:
                m_hat = pm.TruncatedNormal(
                    'm_hat',
                    mu=prior_m_mean,
                    sigma=prior_m_sigma,
                    lower=0.1,
                    shape=(
                        nsp,
                    ))
                print("m_hat is inferred")
            else:
                m_hat = at.as_tensor_variable(m)
                print("m_hat is fixed")

            # For r parameter
            if prior_r_mean is not None and prior_r_sigma is not None:
                r_hat = pm.TruncatedNormal(
                    'r_hat',
                    mu=prior_r_mean,
                    sigma=prior_r_sigma,
                    lower=0.1,
                    shape=(
                        nr,
                    ))
                print("r_hat is inferred")
            else:
                r_hat = at.as_tensor_variable(r)
                print("r_hat is fixed")

            # For K parameter
            if prior_K_mean is not None and prior_K_sigma is not None:
                K_hat = pm.TruncatedNormal(
                    'K_hat',
                    mu=prior_K_mean,
                    sigma=prior_K_sigma,
                    lower=1.0,
                    shape=(
                        nr,
                    ))
                print("K_hat is inferred")
            else:
                K_hat = at.as_tensor_variable(K)
                print("K_hat is fixed")

            # Flatten to read into CRM_inf_func as a single vector
            nr_tensor = at.as_tensor_variable([nr])
            nsp_tensor = at.as_tensor_variable([nsp])

            theta = at.concatenate(
                [nr_tensor, nsp_tensor, tau_hat, w_hat, c_hat.flatten(), m_hat, r_hat, K_hat])

            # Initial conditions for the ODE
            # initial_conditions = np.concatenate([(yobs[0,:nsp]), np.array([10.0, 10.0])])
            # Initial species and resource populations
            y0 = np.concatenate([np.ones(nsp), np.ones(nr)])
            # y0 = np.array([10.0, 10.0, 10.0, 10.0])
            # y0 = np.full(n_states, 10.0)

            # Solve the ODE
            crm_curves = crm_model(y0=y0, theta=theta)

            # Define the log-normal likelihood with log-transformed observed data
            # Y = pm.Lognormal("Y", mu=pm.math.log(crm_curves), sigma=sigma, observed=yobs)
            Y = pm.Lognormal(
                "Y",
                mu=at.log(crm_curves),
                sigma=sigma,
                observed=yobs)

            # For debugging:
            # print if `debug` is set to 'high' or 'low'
            if self.debug in ["high", "low"]:
                initial_values = bayes_model.initial_point()
                print(f"Initial parameter values: {initial_values}")
                print("Shape of tau_hat:", tau_hat.shape.eval())
                print("Shape of w_hat:", w_hat.shape.eval())
                print("Shape of c_hat:", c_hat.shape.eval())
                print("Shape of m_hat:", m_hat.shape.eval())
                print("Shape of r_hat:", r_hat.shape.eval())
                print("Shape of K_hat:", K_hat.shape.eval())
                print("Shape of nr_tensor:", nr_tensor.shape.eval())
                print("Shape of nsp_tensor:", nsp_tensor.shape.eval())
                print("Shape of theta:", theta.shape.eval())
                print("Shape of yobs:", yobs.shape)
                print("Shape of crm_curves:", crm_curves.shape.eval())

            # Sample the posterior
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=True)

        return idata

    def plot_posterior(self, idata):
        """
        Plot the posterior distributions of parameters conditionally based on
        whether they exist in the posterior samples.

        Args:
            idata: InferenceData object containing the posterior samples
        """
        # List of parameter names to check and plot if available
        param_names = ["tau_hat", "w_hat", "c_hat", "m_hat", "r_hat", "K_hat"]

        # Get all available variables in the posterior
        available_vars = list(idata.posterior.data_vars)

        for param in param_names:
            if param in available_vars:
                print(f"Plotting posterior for {param}")

                # Extract the posterior mean for the parameter
                if param == "c_hat":
                    # Special handling for c_hat due to its shape
                    param_np = idata.posterior[param].mean(
                        dim=('chain', 'draw')).values
                    ref_val = param_np.flatten().tolist()
                else:
                    param_np = idata.posterior[param].mean(
                        dim=('chain', 'draw')).values.flatten()
                    ref_val = param_np.tolist()

                # Plot the posterior
                az.plot_posterior(
                    idata,
                    var_names=[param],
                    ref_val=ref_val
                )

                # Save the plot
                plt.savefig(f"plot-posterior-{param}.pdf")
                plt.show()
                plt.close()
            else:
                print(
                    f"Parameter {param} not found in posterior samples, skipping plot.")

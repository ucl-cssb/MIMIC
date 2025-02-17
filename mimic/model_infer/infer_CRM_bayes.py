import arviz as az
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
import pytensor.tensor as at
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
        X (numpy.ndarray): The design matrix
        F (numpy.ndarray): The observed values
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
            X = None,
            F = None,
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

        self.X: Optional[numpy.ndarray] = None
        self.F: Optional[numpy.ndarray] = None

        self.num_species: Optional[int] = None
        self.num_resources: Optional[int] = None

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
        self.prior_tau_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_tau_mean
        self.prior_tau_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_tau_sigma
        self.prior_w_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_w_mean
        self.prior_w_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_w_sigma
        self.prior_c_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_c_mean
        self.prior_c_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_c_sigma
        self.prior_m_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_m_mean
        self.prior_m_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_m_sigma
        self.prior_r_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_r_mean
        self.prior_r_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_r_sigma
        self.prior_K_mean: Optional[Union[int, float, List[Union[int, float]]]] = prior_K_mean
        self.prior_K_sigma: Optional[Union[int, float, List[Union[int, float]]]] = prior_K_sigma
        

        self.parameters = Dict[str,
                              Optional[Union[int,
                                             float,
                                             numpy.ndarray,
                                             str]]] = {
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
            "prior_K_sigma": self.prior_K_sigma}

    def set_parameters(self,
                       X: Optional[numpy.ndarray] = None,
                       F: Optional[numpy.ndarray] = None,
                       num_species: Optional[int] = None,
                       num_resources: Optional[int] = None,
                       tau: Optional[Union[List[float], numpy.ndarray]] = None,
                       w: Optional[Union[List[float], numpy.ndarray]] = None,
                       c: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       m: Optional[Union[List[float], numpy.ndarray]] = None,
                       r: Optional[Union[List[float], numpy.ndarray]] = None,
                       K: Optional[Union[List[float], numpy.ndarray]] = None,
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
            tau (Optional[Union[List[float], numpy.ndarray]]): species_timescales.
            w (Optional[Union[List[float], numpy.ndarray]]): resource_quality.
            c (Optional[Union[List[List[float]], numpy.ndarray]]): relative_preference.
            m (Optional[Union[List[float], numpy.ndarray]]): mortality_rate.
            r (Optional[Union[List[float], numpy.ndarray]]): resource timescale.
            K (Optional[Union[List[float], numpy.ndarray]]): resource capacity.
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

        if X is not None:
            self.X = numpy.array(X)
        if F is not None:
            self.F = numpy.array(F)
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
        self.data = numpy.genfromtxt(file_path, delimiter=',')
        return
    
        # Define the ODE function for CRM model


    def CRM(y,t, p):
        # Unpack parameters from the vector p
        nr = p[0].astype("int32")   # Number of resources
        nsp = p[1].astype("int32")   # Number of species
        tau = p[2:2+nsp]  # Species time scales
        w = p[2+nsp:2+nsp+nr]  # Resource quality
        c_flat = p[2+nsp+nr:2+nsp+nr+(nsp*nr)]  # Flattened resource preferences
        c = c_flat.reshape((nsp, nr))  # Reshape to nsp x nr matrix
        m = p[2+nsp+nr+(nsp*nr):2+(2*nsp)+nr+(nsp*nr)]  # Mortality rates
        r = p[2+(2*nsp)+nr+(nsp*nr):2+(2*nsp)+(2*nr)+(nsp*nr)]  # Resource time scales
        K = p[2+(2*nsp)+(2*nr)+(nsp*nr):2+(2*nsp)+(3*nr)+(nsp*nr)]  # Resource carrying capacities

        # tau = p[:2]  # Species time scales 
        # w = p[2:4]  # Resource quality 
        # c_flat = p[4:8]  # Flattened resource preferences 
        # c = c_flat.reshape((2, 2))  # Reshape to 2x2 matrix
        # m = p[8:10]  # Mortality rates 
        # r = p[10:12]  # Resource time scales 
        # K = p[12:14]  # Resource carrying capacities

        # Separate species (N) and resources (R)
        N = y[:nsp]  # Species populations 
        R = y[nsp:]  # Resource availability 

    
        # Species growth equation (dN)
        growth_term = at.dot(c, w * R)  # Matrix multiplication for species-resource interaction as tensor
        dN = (N / tau) * (growth_term - m)  # Species growth equation

        # Resource consumption equation (dR)
        consumption_term = at.dot(N, c)  # Matrix multiplication for resource consumption by species as tensor
        dR = (1 / (r * K)) * (K - R) * R - consumption_term * R  # Resource consumption equation

        # Combine dN and dR into a single 1D array
        derivatives = [dN[0], dN[1], dR[0], dR[1]]  # Horizontal stacking ensures a 1D array
   
        # Return the derivatives for both species and resources as a single array
        return derivatives  
    



    

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
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")
        print(f"Number of resources: {num_resources}")

        

        # Set up initial conditions and parameters
        nsp = num_species
        nr = num_resources
        n_states = nsp + nr
        n_theta = 2 + (2 * nsp) + (3 * nr) + (nsp * nr)

        # Time points for integration
        times = X

        # Define the DifferentialEquation model
        crm_model = DifferentialEquation(
            func=CRM,  # The ODE function
            times=times,  # Time points for the solution
            n_states=n_states,  # Total state variables (nsp + nr)
            n_theta=n_theta,  # Total number of parameters (2 + tau, w, c, m, r, K)
            t0=0  # Initial time
        )

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters

            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            tau_hat = pm.TruncatedNormal('tau_hat', mu=prior_tau_mean, sigma=prior_tau_sigma, lower=0, shape=(1, num_species)) # species timescales
            w_hat = pm.TruncatedNormal('w_hat', mu=prior_w_mean, sigma=prior_w_sigma, lower=0, shape=(1, num_resources)) # resource quality
            c_hat_vals = pm.TruncatedNormal('c_hat_vals', mu=prior_c_mean, sigma=prior_c_sigma, lower=0, shape=(num_species, num_resources)) # relative resource preferences
            c_hat = pm.Deterministic('c_hat', c_hat_vals)
            m_hat = pm.TruncatedNormal('m_hat', mu=prior_m_mean, sigma=prior_m_sigma, lower=0, shape=(1, num_species)) # mortality rates
            r_hat = pm.TruncatedNormal('r_hat', mu=prior_r_mean, sigma=prior_r_sigma, lower=0, shape=(1, num_resources)) # resource timescales
            K_hat = pm.TruncatedNormal('K_hat', mu=prior_K_mean, sigma=prior_K_sigma, lower=0, shape=(1, num_resources)) # resource carrying capacities

            #print("Shape of tau_hat:", tau_hat.shape.eval())
            #print("Shape of c_hat:", c_hat.shape.eval())


            # Pack parameters into a single vector

            nr_tensor = at.as_tensor_variable([nr]) 
            nsp_tensor = at.as_tensor_variable([nsp])

            theta = at.concatenate([nr_tensor, nsp_tensor, tau_hat, w_hat, c_hat.flatten(), m_hat, r_hat, K_hat])

            # Initial conditions for the ODE
            initial_conditions = np.concatenate([(F[0,:]), np.array([8.0, 12.0])])
            y0 = np.concatenate([np.ones(nsp), np.ones(nr)])  # Initial species and resource populations

            # Solve the ODE
            crm_curves = crm_model(y0=initial_conditions, theta=theta)
            crm_species_curves = crm_curves[1:, :nsp]  # Extract species curves only, excluding first time point
            #print(crm_curves.eval({tau: np.ones(nsp), w: np.ones(nr), c: np.ones((nsp, nr)), m: np.ones(nsp), r: np.ones(nr), K: np.ones(nr)}))

            #print("Adjusted crm_species_curves shape:", crm_species_curves.shape.eval())
            #print("F shape:", F.shape)


            # Define the likelihood
            #Y = pm.Lognormal("Y", mu=at.log(crm_curves), sigma=sigma, observed=F)
            Y = pm.Lognormal("Y", mu=pm.math.log(crm_species_curves), sigma=sigma, observed=F)
            #Y = pm.Normal("Y", mu=crm_species_curves, sigma=sigma, observed=F)
    

            # Sample the posterior
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, progressbar=True)




            # Pack parameters into a single vector
            #theta = at.concatenate([nr, nsp, tau_hat, w_hat, c_hat.flatten(), m_hat, r_hat, K_hat], axis=0)

            # Initial conditions for the ODE
            #initial_conditions = numpy.concatenate([numpy.array([5.0, 2.0]), numpy.array([8.0, 12.0])])
            #y0 = numpy.concatenate([F[0, :num_species], F[0, num_species:]]) # Initial species and resource populations

            # Solve the ODE
            #crm_curves = crm_model(y0=y0, theta=theta)

            # Define the likelihood
            #Y = pm.Lognormal("Y", mu=at.log(crm_curves), sigma=sigma, observed=F)

            #idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, progressbar=True)

        return idata
    


    def plot_posterior(self, idata):

        tau_hat_np = idata.posterior['tau_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        
        w_hat_np = idata.posterior['w_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        
        c_hat_np = idata.posterior['c_hat'].mean(dim=('chain', 'draw')).values

        m_hat_np = idata.posterior['m_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        
        r_hat_np = idata.posterior['r_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        
        K_hat_np = idata.posterior['K_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        

        az.plot_posterior(
            idata,
            var_names=["tau_hat"],
            ref_val=tau_hat_np.tolist())
        plt.savefig("plot-posterior-tau.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["w_hat"],
            ref_val=w_hat_np.tolist())
        plt.savefig("plot-posterior-w.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["c_hat"],
            ref_val=numpy.diag(c_hat_np).tolist())
        plt.savefig("plot-posterior-c.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["m_hat"],
            ref_val=m_hat_np.tolist())
        plt.savefig("plot-posterior-m.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["r_hat"],
            ref_val=r_hat_np.tolist())
        plt.savefig("plot-posterior-r.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["K_hat"],
            ref_val=K_hat_np.tolist())
        plt.savefig("plot-posterior-K.pdf")
        plt.show()
        plt.close()


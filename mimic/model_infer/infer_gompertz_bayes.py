import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as at
from pymc.ode import DifferentialEquation
import os
from typing import Optional, Union, List
import glob
import shutil


from mimic.model_infer.base_infer import BaseInfer

def plot_Gompertz_with_intervals(
        observed_species,
        species_lower,
        species_upper,
        times,
        filename=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot median trajectories
    for i in range(observed_species.shape[1]):
        ax.plot(times, observed_species[:, i],
                label=f'Species {i+1}', linewidth=2)

    # Add confidence ribbons
    for i in range(observed_species.shape[1]):
        ax.fill_between(times, species_lower[:, i], species_upper[:, i],
                        alpha=0.2, color=plt.cm.tab10(i))


    if filename:
        true_data = pd.read_csv(filename)
        true_times = true_data['time'].values

        for i in range(observed_species.shape[1]):
            col_name = f'species_{i+1}'
            if col_name in true_data.columns:
                ax.scatter(
                    true_times,
                    true_data[col_name],
                    marker='o',
                    s=30,
                    color=plt.cm.tab10(i),
                    label=f'True {col_name}')

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Concentration', fontsize=14)
    ax.set_title(
        'Gompertz Model Dynamics with 95% Credible Intervals',
        fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename.split('.')[0]}_with_intervals.png", dpi=300)
    plt.show()


def gompertz_func(y, t, p):
        """
        ODE function for independent Gompertz growth
        Each species grows independently: dN_i/dt = μ_i * N_i * ln(A_i/N_i)
        
        Parameters:
        - y: state vector [N1, N2, ..., N_nsp]
        - p: parameter vector [nsp, mu1, mu2, ..., mu_nsp, A1, A2, ..., A_nsp]
        """
        
        # Unpack parameters
        nsp = p[0].astype("int32")
        mu = p[1:1 + nsp]  # Maximum specific growth rates
        A = p[1 + nsp:1 + 2*nsp]  # carrying capacities
        
        # Species populations
        N = y[:nsp]
        
        # Ensure positive populations and avoid division issues
        eps = 1e-8
        N_safe = at.maximum(N, eps)
        A_safe = at.maximum(A, eps)
        
        # Ensure N doesn't exceed A to avoid negative ln
        N_bounded = at.minimum(N_safe, A_safe - eps)
        
        # Independent Gompertz growth equations
        # dN_i/dt = μ_i * N_i * ln(A_i/N_i)
        ln_term = at.log(A_safe / N_bounded)
        dN = mu * N_bounded * ln_term
        
        # Prevent negative values
        dN = at.where((N < eps) & (dN < 0), 0.0, dN)
        
        return dN




class inferGompertzbayes(BaseInfer):
    """
    Independent Gompertz Growth Model for Bayesian inference.
    
    Each species grows independently with Gompertz dynamics:
    dN_i/dt = μ_i * N_i * ln(A_i/N_i)
    
    Args:
        times (np.ndarray): The times at which observations were made
        yobs (np.ndarray): The observed species values (n_timepoints, nsp)
        num_species (int): The number of species
        
        prior_mu_mean (List): Mean of prior for growth rate parameters (μ)
        prior_mu_sigma (List): Std dev of prior for growth rate parameters
        prior_A_mean (List): Mean of prior for asymptotic maxima (A)  
        prior_A_sigma (List): Std dev of prior for asymptotic maxima
        
        draws, tune, chains, cores: MCMC parameters
    """
    
    def __init__(self,
                 times=None,
                 yobs=None,
                 num_species=None,
                 prior_mu_mean=None,
                 prior_mu_sigma=None,
                 prior_A_mean=None,
                 prior_A_sigma=None):
        
        super().__init__()
        
        self.times = times
        self.yobs = yobs
        self.num_species = num_species
        
        # Default parameter values
        self.mu = None  # Growth rate parameters
        self.A = None   # Asymptotic maxima
        
        # Sampling parameters
        self.draws = None
        self.tune = None
        self.chains = None
        self.cores = None
        
        # Priors
        self.prior_mu_mean = prior_mu_mean
        self.prior_mu_sigma = prior_mu_sigma
        self.prior_A_mean = prior_A_mean
        self.prior_A_sigma = prior_A_sigma
    
    def set_parameters(self,
                      times: Optional[np.ndarray] = None,
                      yobs: Optional[np.ndarray] = None,
                      num_species: Optional[int] = None,
                      mu: Optional[Union[List[float], np.ndarray]] = None,
                      A: Optional[Union[List[float], np.ndarray]] = None,
                      prior_mu_mean: Optional[Union[List[float]]] = None,
                      prior_mu_sigma: Optional[Union[List[float]]] = None,
                      prior_A_mean: Optional[Union[List[float]]] = None,
                      prior_A_sigma: Optional[Union[List[float]]] = None,
                      draws: Optional[int] = None,
                      tune: Optional[int] = None,
                      chains: Optional[int] = None,
                      cores: Optional[int] = None) -> None:
        """
        Updates parameters. Only provided values are updated.
        
        Parameters:
            times: Time points for observations
            yobs: Observed data (species only)
            num_species: Number of species
            mu: Growth rate parameters (if fixing parameters)
            A: Asymptotic maxima (if fixing parameters)
            prior_mu_mean: Prior means for growth rate parameters
            prior_mu_sigma: Prior std devs for growth rate parameters
            prior_A_mean: Prior means for asymptotic maxima
            prior_A_sigma: Prior std devs for asymptotic maxima
            draws, tune, chains, cores: MCMC sampling parameters
        """
        
        if times is not None:
            self.times = np.array(times)
        if yobs is not None:
            self.yobs = np.array(yobs)
        if num_species is not None:
            self.num_species = num_species
        if mu is not None:
            self.mu = mu
        if A is not None:
            self.A = A
            
        # Priors
        if prior_mu_mean is not None:
            self.prior_mu_mean = prior_mu_mean
        if prior_mu_sigma is not None:
            self.prior_mu_sigma = prior_mu_sigma
        if prior_A_mean is not None:
            self.prior_A_mean = prior_A_mean
        if prior_A_sigma is not None:
            self.prior_A_sigma = prior_A_sigma
            
        if draws is not None:
            self.draws = draws
        if tune is not None:
            self.tune = tune
        if chains is not None:
            self.chains = chains
        if cores is not None:
            self.cores = cores
    
    
    
    def run_inference(self):
        """
        Run Bayesian inference for independent logistic growth model
        
        Returns:
            idata: Inference data with posterior samples
        """
        
        if self.times is None or self.yobs is None:
            raise ValueError("times, yobs must both be provided.")
        
        if self.num_species is None:
            raise ValueError("num_species must be provided.")
        
        times = self.times
        yobs = self.yobs
        num_species = self.num_species
        prior_mu_mean = self.prior_mu_mean
        prior_mu_sigma = self.prior_mu_sigma
        prior_A_mean = self.prior_A_mean
        prior_A_sigma = self.prior_A_sigma
        mu = self.mu
        A = self.A

        
        # Print setup info
        print(f"times shape: {self.times.shape}")
        print(f"yobs shape: {self.yobs.shape}")
        print(f"Number of species: {self.num_species}")
        
        # Model dimensions
        nsp = self.num_species
        n_states = nsp  # Only species
        n_theta = 1 + 2*nsp  # nsp + mu(nsp) + A(nsp)
        
        # Define the DifferentialEquation model
        gompertz_model = DifferentialEquation(
            func=gompertz_func,
            times=times,
            n_states=n_states,
            n_theta=n_theta,
            t0=0
        )
        
        bayes_model = pm.Model()
        with bayes_model:
            sigma = pm.HalfNormal('sigma', sigma=0.1, shape=(1,))

            # Conditionally define parameters based on whether priors are provided
            
            # Growth rate parameter priors
            if self.prior_mu_mean is not None and self.prior_mu_sigma is not None:
                mu_hat = pm.TruncatedNormal('mu_hat', mu=self.prior_mu_mean, sigma=self.prior_mu_sigma, lower=0, shape=(nsp,))
                print("mu_hat is inferred")
            else:
                mu_hat = at.as_tensor_variable(mu)  # Set values
                print("mu_hat is fixed")
            
            # Carry capacity parameter priors
            if self.prior_A_mean is not None and self.prior_A_sigma is not None:
                A_hat = pm.TruncatedNormal('A_hat', mu=self.prior_A_mean,  sigma=self.prior_A_sigma, lower=0, shape=(nsp,))
                print("A_hat is inferred")
            else:
                A_hat = at.as_tensor_variable(A)  # Set values
                print("A_hat is fixed")
            
            # Parameter vector
            nsp_tensor = at.as_tensor_variable([nsp])
            theta = at.concatenate([nsp_tensor, mu_hat, A_hat])
            
            # Initial conditions from observed data
            y0 = self.yobs[0, :nsp]  # Use observed initial conditions
            
            print(f"Initial conditions (y0): {y0}")
            
            # Solve the ODE
            gompertz_curves = gompertz_model(y0=y0, theta=theta)
            
            # Likelihood
            Y = pm.Lognormal("Y", mu=at.log(gompertz_curves), sigma=sigma, observed=self.yobs)
            
            # Debug info
            if self.debug in ["high", "low"]:
                print("Independent Gompertz model setup complete")
                print(f"Shape of theta: {theta.shape.eval()}")
                print(f"Shape of yobs: {self.yobs.shape}")
                print(f"Shape of gompertz_curves: {gompertz_curves.shape.eval()}")
            
            # Sample the posterior
            idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains,
                            cores=self.cores, progressbar=True)
        
        return idata
    
    def plot_posterior(self, idata, output_folder="."):
        """
        Plot posterior distributions following CRM style
        
        Parameters:
            idata: Inference data from run_inference()
            output_folder: Directory to save plots
        """
        
        # Get variables to plot
        var_names = ["mu_hat", "A_hat", "sigma"]
        available_vars = [var for var in var_names if var in idata.posterior.data_vars]
        
        if len(available_vars) == 0:
            print("No variables found to plot")
            return
        
        # Count total number of plots needed
        n_plots = 0
        for var_name in available_vars:
            var_data = idata.posterior[var_name]
            if len(var_data.shape) == 3:  # Vector variable
                n_plots += var_data.shape[2]
            else:  # Scalar variable
                n_plots += 1
        
        # Create figure
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        for var_name in available_vars:
            var_data = idata.posterior[var_name]
            
            if len(var_data.shape) == 3:  # Vector variable (like mu_hat, A_hat)
                for i in range(var_data.shape[2]):
                    data_flat = var_data.values[:, :, i].flatten()
                    
                    axes[plot_idx].hist(data_flat, bins=50, alpha=0.7, density=True, 
                                      edgecolor='black')
                    axes[plot_idx].set_xlabel(f'{var_name}[{i}]')
                    axes[plot_idx].set_ylabel('Density')
                    axes[plot_idx].set_title(f'Posterior Distribution of {var_name}[{i}]')
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = np.mean(data_flat)
                    axes[plot_idx].axvline(mean_val, color='red', linestyle='--', 
                                         label=f'Mean: {mean_val:.3f}')
                    axes[plot_idx].legend()
                    
                    plot_idx += 1
            
            else:  # Scalar variable (like sigma)
                data_flat = var_data.values.flatten()
                
                axes[plot_idx].hist(data_flat, bins=50, alpha=0.7, density=True,
                                  edgecolor='black')
                axes[plot_idx].set_xlabel(var_name)
                axes[plot_idx].set_ylabel('Density')
                axes[plot_idx].set_title(f'Posterior Distribution of {var_name}')
                axes[plot_idx].grid(True, alpha=0.3)
                
                # Add mean line
                mean_val = np.mean(data_flat)
                axes[plot_idx].axvline(mean_val, color='red', linestyle='--',
                                     label=f'Mean: {mean_val:.3f}')
                axes[plot_idx].legend()
                
                plot_idx += 1
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"gompertz-plot-posterior-{var_name}.pdf"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved independent Gompertz posterior plot to {plot_filename}")
        
        return plot_filename
    
    
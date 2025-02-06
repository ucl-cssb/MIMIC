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

from mimic.model_infer.base_infer import BaseInfer

import os
from typing import Optional, Union, List, Dict, Any


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Used in examples-Stein.ipynb
def plot_params(mu_h, M_h, e_h, nsp):
    print("\ninferred params:")
    print("mu_hat/mu:")
    print(np.array(mu_h))
    print("\nM_hat/M:")
    print(np.round(np.array(M_h), decimals=2))
    print("e_hat/e:")
    print(np.array(e_h))

    # plot the params
    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 3, 1)
    plt.stem(np.arange(0, nsp, dtype="int32"), np.array(mu_h), markerfmt="D")

    plt.subplot(1, 3, 2)
    plt.stem(np.arange(0, nsp * nsp), np.array(M_h).flatten(), markerfmt="D")

    plt.subplot(1, 3, 3)
    plt.stem(np.arange(0, nsp), np.array(e_h), markerfmt="D")


class infergLVbayes(BaseInfer):
    """
    bayes_gLV class for Bayesian inference of gLV models without shrinkage priors

    Args:
        X (np.ndarray): The design matrix
        F (np.ndarray): The observed values
        mu (np.ndarray): The growth rates matrix
        M (np.ndarray): The interaction matrix


    Methods:


    Returns:
        None
    """

    def __init__(self,
                 X=None,
                 F=None,
                 prior_mu_mean=None,
                 prior_mu_sigma=None,
                 prior_Mii_mean=None,
                 prior_Mii_sigma=None,
                 prior_Mij_sigma=None
                 ):

        # self.data = data  # data to do inference on
        self.X: Optional[np.ndarray] = X
        self.F: Optional[np.ndarray] = F
        self.mu: Optional[Union[int, float]] = None
        self.M: Optional[Union[int, float]] = None
        self.prior_mu_mean: Optional[Union[int, float,
                                           List[Union[int, float]]]] = prior_mu_mean
        self.prior_mu_sigma: Optional[Union[int, float,
                                            List[Union[int, float]]]] = prior_mu_sigma
        self.prior_Mii_mean: Optional[Union[int, float,
                                            List[Union[int, float]]]] = prior_Mii_mean
        self.prior_Mii_sigma: Optional[Union[int, float,
                                             List[Union[int, float]]]] = prior_Mii_sigma
        self.prior_Mij_sigma: Optional[Union[int, float,
                                             List[Union[int, float]]]] = prior_Mij_sigma
        self.prior_eps_mean: Optional[Union[int,
                                            float, List[Union[int, float]]]] = None
        self.prior_eps_sigma: Optional[Union[int,
                                             float, List[Union[int, float]]]] = None
        self.draws: Optional[int] = None
        self.tune: Optional[int] = None
        self.chains: Optional[int] = None
        self.cores: Optional[int] = None
        self.DA: Optional[int] = None
        self.DA0: Optional[Union[int, float]] = None
        self.N: Optional[int] = None
        self.noise_stddev: Optional[Union[int, float]] = None
        self.epsilon: Optional[Union[int, float,
                                     List[Union[int, float]]]] = None,
        self.sim_glv: Optional[str] = None

        # Calculate DA0 if F is not None and DA0 is not set
        if self.DA0 is None and self.F is not None:
            self.DA0 = self.calculate_DA0(self.F.shape[1])

        self.parameters: Dict[str,
                              Optional[Union[int,
                                             float,
                                             np.ndarray,
                                             str]]] = {"prior_mu_mean": self.prior_mu_mean,
                                                       "prior_mu_sigma": self.prior_mu_sigma,
                                                       "prior_Mii_mean": self.prior_Mii_mean,
                                                       "prior_Mii_sigma": self.prior_Mii_sigma,
                                                       "prior_Mij_sigma": self.prior_Mij_sigma,
                                                       "prior_eps_mean": self.prior_eps_mean,
                                                       "prior_eps_sigma": self.prior_eps_sigma,
                                                       "DA": self.DA,
                                                       "DA0": self.DA0,
                                                       "N": self.N,
                                                       "noise_stddev": self.noise_stddev}

    def set_parameters(self,
                       X: Optional[np.ndarray] = None,
                       F: Optional[np.ndarray] = None,
                       prior_mu_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_mu_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_Mii_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_Mii_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_Mij_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_eps_mean: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       prior_eps_sigma: Optional[Union[int, float, List[Union[int, float]]]] = None,
                       draws: Optional[int] = None,
                       tune: Optional[int] = None,
                       chains: Optional[int] = None,
                       cores: Optional[int] = None,
                       DA: Optional[Union[int, float]] = None,
                       DA0: Optional[Union[int, float]] = None,
                       N: Optional[int] = None,
                       noise_stddev: Optional[Union[int, float]] = None) -> None:

        if X is not None:
            self.X = np.array(X)
        if F is not None:
            self.F = np.array(F)
        if prior_mu_mean is not None:
            self.prior_mu_mean = prior_mu_mean
        if prior_mu_sigma is not None:
            self.prior_mu_sigma = prior_mu_sigma
        if prior_Mii_mean is not None:
            self.prior_Mii_mean = prior_Mii_mean
        if prior_Mii_sigma is not None:
            self.prior_Mii_sigma = prior_Mii_sigma
        if prior_Mij_sigma is not None:
            self.prior_Mij_sigma = prior_Mij_sigma
        if prior_eps_mean is not None:
            self.prior_eps_mean = prior_eps_mean
        if prior_eps_sigma is not None:
            self.prior_eps_sigma = prior_eps_sigma
        if draws is not None:
            self.draws = draws
        if tune is not None:
            self.tune = tune
        if chains is not None:
            self.chains = chains
        if cores is not None:
            self.cores = cores
        if DA is not None:
            self.DA = DA
        if DA0 is not None:
            self.DA0 = DA0 if DA0 is not None else (
                self.calculate_DA0(F.shape[1]) if F is not None else None)
        if N is not None:
            self.N = N
        if noise_stddev is not None:
            self.noise_stddev = noise_stddev

        self.parameters = {
            "X": self.X,
            "F": self.F,
            "prior_mu_mean": self.prior_mu_mean,
            "prior_mu_sigma": self.prior_mu_sigma,
            "prior_Mii_mean": self.prior_Mii_mean,
            "prior_Mii_sigma": self.prior_Mii_sigma,
            "prior_Mij_sigma": self.prior_Mij_sigma,
            "prior_eps_mean": self.prior_eps_mean,
            "prior_eps_sigma": self.prior_eps_sigma,
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "cores": self.cores,
            "DA": self.DA,
            "DA0": self.DA0,
            "calculate_DA0": self.calculate_DA0,
            "N": self.N,
            "noise_stddev": self.noise_stddev}

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

    def calculate_DA0(self, num_species, proportion=0.15):
        total_off_diagonal_elements = num_species * (num_species - 1)
        expected_non_zero_elements = total_off_diagonal_elements * proportion
        DA0 = int(round(expected_non_zero_elements))
        return max(DA0, 1)

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
        prior_mu_mean = self.prior_mu_mean
        prior_mu_sigma = self.prior_mu_sigma
        prior_Mii_mean = self.prior_Mii_mean
        prior_Mii_sigma = self.prior_Mii_sigma
        prior_Mij_sigma = self.prior_Mij_sigma
        draws = self.draws
        tune = self.tune
        chains = self.chains
        cores = self.cores

        num_species = F.shape[1]

        # Print shapes to ensure data is correct
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  #
            # A separate sigma for each response
            sigma = pm.HalfNormal(
                'sigma', sigma=1, shape=(
                    1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal(
                'mu_hat',
                mu=prior_mu_mean,
                sigma=prior_mu_sigma,
                lower=0,
                shape=(
                    1,
                    num_species))

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.TruncatedNormal(
                'M_ii_hat_p',
                mu=prior_Mii_mean,
                sigma=prior_Mii_sigma,
                lower=0,
                shape=(
                    num_species,
                ))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ij is unconstrained
            M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=prior_Mij_sigma, shape=(
                num_species, num_species - 1))  # different shape for off-diagonal

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((num_species, num_species))
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(
                num_species), at.arange(num_species)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(num_species)[:, None], np.delete(
                np.arange(num_species), -1)], M_ij_hat)  # set off-diagonal
            print(M_hat_vals)

            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat, mu_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # For debugging:
            # print if `debug` is set to 'high' or 'low'
            if self.debug in ["high", "low"]:
                initial_values = bayes_model.initial_point()
                print(f"Initial parameter values: {initial_values}")

            # As tensor objects are symbolic, if needed print using .eval()
            # eg
            # print(f"mu_hat: {mu_hat.eval()}")

            # initial_values = bayes_model.initial_point()
            # print(f"Initial parameter values: {initial_values}")

            # Posterior distribution
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=True)

        return idata

    def run_inference_shrinkage(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model with Horseshoe prior for shrinkage

        Returns:
            idata: The posterior inference data

        """

        X = self.X
        F = self.F
        prior_mu_mean = self.prior_mu_mean
        prior_mu_sigma = self.prior_mu_sigma
        prior_Mii_mean = self.prior_Mii_mean
        prior_Mii_sigma = self.prior_Mii_sigma
        prior_Mij_sigma = self.prior_Mij_sigma
        DA = self.DA
        DA0 = self.DA0
        N = self.N
        noise_stddev = self.noise_stddev
        draws = self.draws
        tune = self.tune
        chains = self.chains
        cores = self.cores

        # Print the values to verify
        # print(f"DA: {DA}, DA0: {DA0}, N: {N}, noise_stddev: {noise_stddev}")

        num_species = F.shape[1]

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  #
            # A separate sigma for each response
            sigma = pm.HalfNormal(
                'sigma', sigma=1, shape=(
                    1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal(
                'mu_hat',
                mu=prior_mu_mean,
                sigma=prior_mu_sigma,
                lower=0,
                shape=(
                    1,
                    num_species))

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.TruncatedNormal(
                'M_ii_hat_p',
                mu=prior_Mii_mean,
                sigma=prior_Mii_sigma,
                lower=0,
                shape=(
                    num_species,
                ))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ii_hat = pm.TruncatedNormal('M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(num_species,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy(
                "lam", beta=1, shape=(
                    num_species, num_species - 1))
            M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=tau *
                                 lam *
                                 at.sqrt(c2 /
                                         (c2 +
                                          tau ** 2 *
                                          lam ** 2)), shape=(num_species, num_species -
                                                             1))
            # M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=prior_Mij_sigma,
            # shape=(num_species, num_species - 1))  # different shape for
            # off-diagonal

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((num_species, num_species))
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(
                num_species), at.arange(num_species)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(num_species)[:, None], np.delete(
                np.arange(num_species), -1)], M_ij_hat)  # set off-diagonal

            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat, mu_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # For debugging:

            # As tensor objects are symbolic, if needed print using .eval()
            # eg
            # print(f"mu_hat: {mu_hat.eval()}")

            # initial_values = bayes_model.initial_point()
            # print(f"Initial parameter values: {initial_values}")

            # Posterior distribution
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores)

        return idata

    def run_inference_shrinkage_pert(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model with Horseshoe prior for shrinkage

        Returns:
            idata: The posterior inference data
        """

        X = self.X
        F = self.F
        prior_mu_mean = self.prior_mu_mean
        prior_mu_sigma = self.prior_mu_sigma
        prior_Mii_mean = self.prior_Mii_mean
        prior_Mii_sigma = self.prior_Mii_sigma
        prior_Mij_sigma = self.prior_Mij_sigma
        prior_eps_mean = self.prior_eps_mean
        prior_eps_sigma = self.prior_eps_sigma
        DA = self.DA
        DA0 = self.DA0
        N = self.N
        noise_stddev = self.noise_stddev
        draws = self.draws
        tune = self.tune
        chains = self.chains
        cores = self.cores

        # Print the values to debug
        # print(f"DA: {DA}, DA0: {DA0}, N: {N}, noise_stddev: {noise_stddev}")

        num_species = F.shape[1]

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  #
            # A separate sigma for each response
            sigma = pm.HalfNormal(
                'sigma', sigma=1, shape=(
                    1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal(
                'mu_hat',
                mu=prior_mu_mean,
                sigma=prior_mu_sigma,
                lower=0,
                shape=(
                    1,
                    num_species))

            epsilon_hat = pm.Normal(
                'epsilon_hat',
                mu=prior_eps_mean,
                sigma=prior_eps_sigma,
                shape=(
                    1,
                    num_species))

            # Set constraints for horseshoe prior
            # M_ij is is unconstrained but placed under horseshoe prior to
            # apply to sigma for M_ij

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.TruncatedNormal(
                'M_ii_hat_p',
                mu=prior_Mii_mean,
                sigma=prior_Mii_sigma,
                lower=0,
                shape=(
                    num_species,
                ))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ii_hat = pm.TruncatedNormal('M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(num_species,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy(
                "lam", beta=1, shape=(
                    num_species, num_species - 1))
            M_ij_hat = pm.Normal('M_ij_hat', mu=prior_Mij_sigma, sigma=tau * lam *
                                 at.sqrt(c2 / (c2 + tau ** 2 * lam ** 2)),
                                 shape=(num_species,
                                        num_species - 1))
            # M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=prior_Mij_sigma,
            # shape=(num_species, num_species - 1))  # different shape for
            # off-diagonal

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((num_species, num_species))
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(
                num_species), at.arange(num_species)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(num_species)[:, None], np.delete(
                np.arange(num_species), -1)], M_ij_hat)  # set off-diagonal

            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            # model_mean = pm.math.dot(X, pm.math.concatenate([M_hat, mu_hat], axis=0))
            model_mean = pm.math.dot(X, pm.math.concatenate(
                [M_hat, mu_hat, epsilon_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # For debugging:

            # As tensor objects are symbolic, if needed print using .eval()
            # eg
            # print(f"mu_hat: {mu_hat.eval()}")

            # initial_values = bayes_model.initial_point()
            # print(f"Initial parameter values: {initial_values}")

            # Posterior distribution
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores)

            # Assemble posterior values for mu and M for plotting and
            # assessment
        # mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        # M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        # Plot and save posterior results
        # self.plot_posterior_pert(idata, mu_hat_np, M_hat_np, epsilon)

        return idata

    def plot_posterior(self, idata):

        mu_hat_np = idata.posterior['mu_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        az.plot_posterior(
            idata,
            var_names=["mu_hat"],
            ref_val=mu_hat_np.tolist()
        )
        plt.savefig("plot-posterior-mu.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["M_ii_hat"],
            ref_val=np.diag(M_hat_np).tolist()
        )
        plt.savefig("plot-posterior-Mii.pdf")
        plt.show()
        plt.close()

        mask = ~np.eye(M_hat_np.shape[0], dtype=bool)
        M_ij = M_hat_np[mask]
        az.plot_posterior(
            idata,
            var_names=["M_ij_hat"],
            ref_val=M_ij.flatten().tolist()
        )
        plt.savefig("plot-posterior-Mij.pdf")
        plt.show()
        plt.close()

    def plot_posterior_pert(self, idata):
        mu_hat_np = idata.posterior['mu_hat'].mean(
            dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values
        epsilon_hat_np = idata.posterior['epsilon_hat'].mean(
            dim=('chain', 'draw')).values

        az.plot_posterior(
            idata,
            var_names=["mu_hat"],
            ref_val=mu_hat_np.tolist())
        plt.savefig("plot-posterior-mu.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["M_ii_hat"],
            ref_val=np.diag(M_hat_np).tolist())
        plt.savefig("plot-posterior-Mii.pdf")
        plt.show()
        plt.close()

        mask = ~np.eye(M_hat_np.shape[0], dtype=bool)
        M_ij = M_hat_np[mask]
        az.plot_posterior(
            idata,
            var_names=["M_ij_hat"],
            ref_val=M_ij.flatten().tolist())
        plt.savefig("plot-posterior-Mij.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(
            idata,
            var_names=["epsilon_hat"],
            ref_val=epsilon_hat_np.flatten().tolist())
        plt.savefig("plot-posterior-eps.pdf")
        plt.show()
        plt.close()

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


def param_data_compare(
        idata,
        F,
        mu,
        M,
        times,
        yobs,
        init_species_start,
        sim_gLV_class):
    # az.to_netcdf(idata, 'model_posterior.nc')
    # Compare model parameters to the data

    num_species = F.shape[1]
    print(num_species)
    init_species = init_species_start * np.ones(num_species)
    # init_species = 0.01 * np.ones(num_species)

    print(idata.posterior["M_hat"].values.shape)

    print(idata.posterior["mu_hat"].values.shape)

    # # get median posterior values
    M_h = np.median(idata.posterior["M_hat"].values, axis=(0, 1))

    mu_h = np.median(idata.posterior["mu_hat"].values, axis=(0, 1))
    mu_h = mu_h.flatten()

    infer_h = infergLVbayes(M_h)
    matrix = infer_h.plot_interaction_matrix(M=M, M_h=M_h)

    # mu_h = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
    # M_h= idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

    predictor = sim_gLV(num_species=num_species, M=M_h, mu=mu_h)
    yobs_h, _, _, _, _ = predictor.simulate(
        times=times, init_species=init_species)

    plot_fit_gLV(yobs, yobs_h, times)
    compare_params(mu=(mu, mu_h), M=(M, M_h))


def curve_compare(idata, F, times, yobs, init_species_start, sim_gLV_class):
    # Compare model parameters to the data
    num_species = F.shape[1]
    # init_species = 10 * np.ones(num_species)
    # init_species = 0.01 * np.ones(num_species)
    init_species = init_species_start * np.ones(num_species)

    # print(idata.posterior["M_hat"].values.shape)

    # # get median posterior values
    M_h = np.median(idata.posterior["M_hat"].values, axis=(0, 1)).T

    mu_h = np.median(idata.posterior["mu_hat"].values, axis=(0, 1))
    mu_h = mu_h.flatten()

    # mu_h = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
    # M_h= idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

    predictor = sim_gLV(num_species=num_species,
                        M=M_h.T,
                        mu=mu_h
                        )
    yobs_h, _, _, _, _ = predictor.simulate(
        times=times, init_species=init_species)

    plot_fit_gLV(yobs, yobs_h, times)


def param_data_compare_pert(
        idata,
        F,
        mu,
        M,
        epsilon,
        num_perturbations,
        times,
        yobs,
        init_species_start,
        sim_gLV_class):
    # az.to_netcdf(idata, 'model_posterior.nc')
    # Compare model parameters to the data
    num_species = F.shape[1]
    init_species = init_species_start * np.ones(num_species)
    # init_species = 0.01 * np.ones(num_species)

    print(idata.posterior["M_hat"].values.shape)

    print(idata.posterior["mu_hat"].values.shape)

    # # get median posterior values
    M_h = np.median(idata.posterior["M_hat"].values, axis=(0, 1))

    mu_h = np.median(idata.posterior["mu_hat"].values, axis=(0, 1))
    mu_h = mu_h.flatten()

    e_h = np.median(idata.posterior["epsilon_hat"].values, axis=(0, 1))
    # reshape e_h so it is the same shape as epsilon
    e_h = e_h.reshape(epsilon.shape)

    infer_h = infergLVbayes(M_h)
    matrix = infer_h.plot_interaction_matrix(M=M, M_h=M_h)

    # mu_h = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
    # M_h= idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

    predictor = sim_gLV(num_species=num_species, M=M_h, mu=mu_h, epsilon=e_h)
    simulator = sim_gLV(
        num_species=num_species,
        num_perturbations=num_perturbations,
        M=M,
        mu=mu,
        epsilon=epsilon)

    yobs, init_species, mu, M, _ = simulator.simulate(
        times=times, init_species=init_species, u=pert_fn)
    yobs_h, _, _, _, _ = predictor.simulate(
        times=times, init_species=init_species, u=pert_fn)

    plot_fit_gLV(yobs, yobs_h, times)

    # compare median posterior values to true values
    compare_params(mu=(mu, mu_h), M=(M, M_h), e=(epsilon, e_h))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as at
import pickle
import cloudpickle


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class infergLVbayes:
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

    def __init__(self, X=None, F=None, mu=None, M=None, M_h=None):
        # self.data = data  # data to do inference on
        self.X = X
        self.F = F
        self.mu = mu
        self.M = M
        self.M_h = M_h
        # self.X: Optional[np.ndarray] = None

        # import data from a .csv file

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

    def run_bayes_gLV(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model

        Returns:
            idata: The posterior inference data


        """

        if self.X is None or self.F is None or self.mu is None or self.M is None:
            raise ValueError("X, F, mu, and M must all be provided.")

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(5,))  # A separate
            # sigma for each response
            sigma = pm.HalfNormal(
                'sigma', sigma=1, shape=(
                    1,))  # Same sigma for all responses

            mu_hat = pm.HalfNormal('mu_hat', sigma=1, shape=(1, 5))

            # M_hat = pm.Normal('M_hat', mu=0, sigma=0.1, shape=(5, 5))  #
            # tighter prior for the slopes

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.HalfNormal('M_ii_hat_p', sigma=0.1, shape=(5,))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ij is unconstrained
            M_ij_hat = pm.Normal(
                'M_ij_hat', mu=0, sigma=0.1, shape=(
                    5, 4))  # different shape for off-diagonal

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((5, 5))
            M_hat_vals = at.set_subtensor(
                M_hat_vals[at.arange(5), at.arange(5)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(5)[:, None], np.delete(
                np.arange(5), -1)], M_ij_hat)  # set off-diagonal

            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat_vals, mu_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # Posterior distribution
            idata = pm.sample(1000, tune=1000, chains=4, cores=4)

        # Plot and save posterior results
        self.plot_posterior(idata, mu, M)

        return idata

        # print summary
        summary = az.summary(
            idata,
            var_names=[
                "mu_hat",
                "M_ii_hat",
                "M_ij_hat",
                "M_hat",
                "sigma"])
        print(summary[["mean", "sd", "r_hat"]])

    def plot_posterior(self, idata, mu, M):
        """
                Plots the posterior distributions and saves the plots to files.

                Args:
                    idata: The posterior inference data.
                    mu (np.ndarray): The growth rates matrix.
                    M (np.ndarray): The interaction matrix.
                """
        az.plot_posterior(
            idata,
            var_names=["mu_hat"],
            ref_val=mu.flatten().tolist())
        plt.savefig("plot-posterior-mu.pdf")

        az.plot_posterior(
            idata,
            var_names=["M_ii_hat"],
            ref_val=np.diag(M).tolist())
        plt.savefig("plot-posterior-Mii.pdf")

        mask = ~np.eye(M.shape[0], dtype=bool)
        M_ij = M[mask]
        az.plot_posterior(
            idata,
            var_names=["M_ij_hat"],
            ref_val=M_ij.flatten().tolist())
        plt.savefig("plot-posterior-Mij.pdf")

      #  # Write posterior samples to file
      #  az.to_netcdf(idata, 'model_posterior.nc')

      #  # read in posterior samples
      #  idata = az.from_netcdf('model_posterior.nc')

      #  az.plot_posterior(idata, var_names=["mu_hat"], ref_val=mu.flatten().tolist())
      #  plt.savefig("plot-posterior-mu.pdf")

      #  az.plot_posterior(idata, var_names=["M_ii_hat"], ref_val=np.diag(M).tolist())
      #  plt.savefig("plot-posterior-Mii.pdf")

      #  mask = ~np.eye(M.shape[0], dtype=bool)
      #  M_ij = M[mask]
      #  az.plot_posterior(idata, var_names=["M_ij_hat"], ref_val=M_ij.flatten().tolist())
      #  plt.savefig("plot-posterior-Mij.pdf")

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

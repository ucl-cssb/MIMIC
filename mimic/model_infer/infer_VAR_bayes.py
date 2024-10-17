import os
from typing import Optional, Union, List, Dict, Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
import seaborn as sns

from mimic.model_infer.base_infer import BaseInfer


class infer_VAR(BaseInfer):
    """
    infer_VAR class for performing inference on VAR models.

    This class provides methods for importing data, running inference on the VAR model, and performing posterior sampling and analysis.

    Args:
        data (numpy.ndarray): The data to perform inference on.

    Methods:
        run_inference():
            Runs the inference process for the VAR model.

        run_inference_large():
            Runs large-scale inference for VAR model.

        run_inference_xs():
            Runs the inference process for the VAR model with metabolite data.

        run_inference_large_xs():
            Runs large-scale inference for the VAR model with metabolite data.

    Returns:
        None
    """

    def __init__(
            self,
            data=None,
            coefficients=None,
            intercepts=None,
            covariance_matrix=None,
            dataS=None):
        super().__init__()  # Call base class constructor
        self.data = self._validate_data(data)
        self.dataS = self._validate_data(dataS) if dataS is not None else None
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.covariance_matrix = covariance_matrix

        # Initialize attributes for storing data and filenames
        self.last_trace = None
        self.last_data = None
        self.last_data_filename = None
        self.last_netcdf_filename = None

    def set_parameters(
            self,
            data: Optional[Union[np.ndarray,
                                 pd.DataFrame, list, tuple]] = None,
            coefficients: Optional[Union[np.ndarray, list]] = None,
            intercepts: Optional[Union[np.ndarray, list]] = None,
            covariance_matrix: Optional[Union[np.ndarray, list]] = None,
            dataS: Optional[Union[np.ndarray,
                                  pd.DataFrame, list, tuple]] = None,
            priors: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the parameters for the infer_VAR instance.

        Allows optional specification of all model parameters. Parameters not provided (None) are left unchanged.

        Parameters:
            data (Optional[Union[np.ndarray, pd.DataFrame, list, tuple]]): The data to perform inference on.
            coefficients (Optional[Union[np.ndarray, list]]): Coefficients of the VAR model.
            intercepts (Optional[Union[np.ndarray, list]]): Intercepts of the VAR model.
            covariance_matrix (Optional[Union[np.ndarray, list]]): Covariance matrix of the VAR model.
            dataS (Optional[Union[np.ndarray, pd.DataFrame, list, tuple]]): The secondary data (e.g., metabolite data).
            priors (Optional[Dict[str, Any]]): A dictionary of prior distributions.
        """
        if data is not None:
            self.data = self._validate_data(data)
        if dataS is not None:
            self.dataS = self._validate_data(dataS)
        if coefficients is not None:
            coefficients = np.array(coefficients)
            if coefficients.shape[0] != coefficients.shape[1]:
                raise ValueError("Coefficients matrix must be square.")
            if self.data is not None and coefficients.shape[0] != self.data.shape[1]:
                raise ValueError(
                    "Coefficients matrix dimensions must match the number of variables in the data.")
            self.coefficients = coefficients
            self.priors['coefficients'] = self.coefficients
        if intercepts is not None:
            intercepts = np.array(intercepts)
            if self.data is not None and intercepts.shape[0] != self.data.shape[1]:
                raise ValueError(
                    "Intercepts must match the number of variables in the data.")
            self.intercepts = intercepts
            self.priors['intercepts'] = self.intercepts
        if covariance_matrix is not None:
            covariance_matrix = np.array(covariance_matrix)
            if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
                raise ValueError("Covariance matrix must be square.")
            if self.data is not None and covariance_matrix.shape[0] != self.data.shape[1]:
                raise ValueError(
                    "Covariance matrix dimensions must match the number of variables in the data.")
            self.covariance_matrix = covariance_matrix
            self.priors['covariance_matrix'] = self.covariance_matrix
        if priors is not None:
            self.set_priors(priors)

    def run_inference(self, **kwargs) -> None:
        """
        Runs the inference process for the VAR model.

        Returns:
        None
        """
        samples = kwargs.get('samples', 2000)
        tune = kwargs.get('tune', 1000)
        cores = kwargs.get('cores', 2)

        if self.data is None:
            print("Error: No data to perform inference on.")
            return
        data = self.data

        # Check the dimensions of the data
        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D array")
        if data.shape[0] < 2:
            raise ValueError("Data must have at least 2 time points")
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 variables")

        # Set the dimension of the VAR model
        dim = data.shape[1]

        # PyMC3 model
        # Set priors if provided, else default to zero mean and unit variance
        x0_prior_mu = self.priors.get('intercepts', self.intercepts.flatten(
        )) if self.intercepts is not None else np.zeros(dim)
        A_prior_mu = self.priors.get(
            'coefficients',
            self.coefficients) if self.coefficients is not None else np.zeros(
            (dim,
             dim))
        noise_cov_prior = self.priors.get(
            'covariance_matrix',
            self.covariance_matrix) if self.covariance_matrix is not None else None

        # PyMC3 model
        with pm.Model() as var_model:
            # Priors for x0 and sigma
            # QUESTION: should the sigma be the noise_stddev from the
            # parameters.json file?
            x0 = pm.Normal('x0', mu=x0_prior_mu, sigma=1, shape=(dim,))
            A = pm.Normal('A', mu=A_prior_mu, sigma=1, shape=(dim, dim))

            # Priors for coefficients with LKJ prior
            # packed_L = pm.LKJCholeskyCov('packed_L', n=dim, eta=2.0, sd_dist=pm.HalfCauchy.dist(2.5))
            # L = pm.expand_packed_triangular(dim, packed_L)
            # coefficients = pm.MvNormal('coefficients', mu=0, chol=L, shape=(dim, dim))

            # If noise covariance is provided, use it as a prior
            if noise_cov_prior is not None:
                noise_chol = np.linalg.cholesky(noise_cov_prior)
                # noise_chol = pm.Deterministic("noise_chol", noise_chol)

            # Otherwise, use LKJ prior
            else:
                noise_chol, _, _ = pm.LKJCholeskyCov(
                    "noise_chol", eta=1.0, n=dim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

            # VAR(1) process likelihood
            x0_obs = data[0, :].copy().reshape(-1)
            mu = x0[:, np.newaxis] + pm.math.dot(A, data[:-1, :].T)

            # obs_chol = np.diag(np.full(dim,sigma))

            # *pm.Normal('likelihood_0', mu=x0, sigma=1.0, observed=x0_obs)

            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            # FIXME: #38 make these arguments specifiable in the
            # parameters.json file file
            trace = pm.sample(draws=samples, tune=tune, cores=cores)

        # Plotting the posterior distributions
        # pm.plot_posterior(trace, var_names=['x0', 'A'])

        # Extracting the posterior means for initial values and coefficients
        # posterior_means = {
        #    'x0': np.mean(trace['x0'], axis=0),
        #    'A': np.mean(trace['A'], axis=0)
        #

        # Store data directly in object attributes
        self.last_trace = trace
        self.last_data = (data,)

        # print if `debug` is set to 'high' or 'low'
        if self.debug in ["high", "low"]:
            print(az.summary(trace, var_names=["x0", "A"]))

        az.plot_posterior(trace, var_names=["x0", "A"])
        plt.savefig("posterior_plot.pdf")
        # Save results using the _save_results method
        self._save_results(trace, (data,), method='default')

    def run_inference_large(self, samples=4000, tune=2000, cores=4) -> None:
        """
        Run large-scale inference for VAR model.

        This function generates VAR model data, fits a Bayesian VAR model using PyMC3,
        and performs posterior sampling and analysis.

        Returns:
            None
        """
        if self.data is None:
            print("Error: No data to perform inference on.")
            return
        data = self.data
        # Check the dimensions of the data
        ndim = data.shape[1]

        # Params for shrinkage
        D = ndim**2
        D0 = 2
        N = data.shape[0]

        # Set priors if provided, else default to zero mean and unit variance
        x0_prior_mu = self.priors.get('intercepts', self.intercepts.flatten(
        )) if self.intercepts is not None else np.zeros(ndim)
        A_prior_mu = self.priors.get(
            'coefficients',
            self.coefficients) if self.coefficients is not None else np.zeros(
            (ndim,
             ndim))
        noise_cov_prior = self.priors.get(
            'covariance_matrix',
            self.covariance_matrix) if self.covariance_matrix is not None else None

        # create and fit PyMC model
        with pm.Model() as var_model:
            noise_stddev = pm.HalfNormal("noise_stddev", 25)
            x0 = pm.Normal('x0', mu=x0_prior_mu, sigma=0.001, shape=(ndim,))

            # Regularised horse shoe prior
            tau0 = (D0 / (D - D0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 8)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy("lam", beta=1, shape=(ndim, ndim))
            A = pm.Normal('A', mu=A_prior_mu, sigma=tau * lam * \
                          at.sqrt(c2 / (c2 + tau**2 * lam**2)), shape=(ndim, ndim))

            # If noise covariance is provided, use it as a prior
            if noise_cov_prior is not None:
                noise_chol = np.linalg.cholesky(noise_cov_prior)

            # Otherwise, use LKJ prior
            else:
                noise_chol, _, _ = pm.LKJCholeskyCov(
                    "noise_chol", eta=1.0, n=ndim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

            # VAR(1) process likelihood
            mu = x0[:, np.newaxis] + pm.math.dot(A, data[:-1, :].T)
            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            trace = pm.sample(draws=samples, tune=tune, cores=cores)

        # print if `debug` is set to 'high' or 'low'
        if self.debug in ["high", "low"]:
            print(az.summary(trace, var_names=["A"]))

        # Plotting the posterior distributions
        az.plot_posterior(trace, var_names=[
            "A"])
        plt.savefig("plot-posterior.pdf")

        # Store data directly in object attributes
        self.last_trace = trace
        self.last_data = (data,)
        # Save results to unique filenames
        self._save_results(trace, data, method='large')

    def run_inference_xs(self, samples=2000, tune=1000, cores=2) -> None:
        """
        Runs the inference process for the VAR model with metabolite data.

        Returns:
        None
        """
        if self.dataS is None:
            raise ValueError(
                "Metabolite data is missing. Please provide dataS.")

        if self.data is None:
            raise ValueError(
                "Abundance data is missing. Please provide dataX.")

        dataX = self.data
        dataS = self.dataS
        nX = dataX.shape[1]
        nS = dataS.shape[1]

        # Set priors if provided, else default to zero mean and unit variance
        x0_prior_mu = self.priors.get('intercepts', self.intercepts.flatten(
        )) if self.intercepts is not None else np.zeros(nX)
        A_prior_mu = self.priors.get(
            'coefficients',
            self.coefficients) if self.coefficients is not None else np.zeros(
            (nX,
             nX))
        noise_cov_prior = self.priors.get(
            'covariance_matrix',
            self.covariance_matrix) if self.covariance_matrix is not None else None

        # PyMC3 model
        with pm.Model() as var_model:
            X0h = pm.Normal('X0h', mu=x0_prior_mu, sigma=1, shape=(nX,))
            S0h = pm.Normal('S0h', mu=0, sigma=1, shape=(nS,))
            Ah = pm.Normal('Ah', mu=A_prior_mu, sigma=1, shape=(nX, nX))
            Bh = pm.Normal('Bh', mu=0, sigma=1, shape=(nS, nX))

            if noise_cov_prior is not None:
                sigma = np.linalg.cholesky(noise_cov_prior)
            else:
                sigma = pm.HalfNormal('sigma', sigma=1, shape=(nX + nS))

            data = np.concatenate((dataX, dataS), axis=1)

            muX = pm.Deterministic('muX', pm.math.dot(Ah, dataX[:-1, :].T))
            muS = pm.math.dot(Bh, muX)
            muXs = muX[:, 1:]
            muSs = muS[:, :-1]
            mu = pm.math.concatenate((muXs, muSs), axis=0)

            likelihood = pm.Normal('likelihood', mu=mu.T,
                                   sigma=sigma, observed=data[2:, :])

        with var_model:
            idata = pm.sample(draws=samples, tune=tune, cores=cores)

        # print if `debug` is set to 'high' or 'low'
        if self.debug in ["high", "low"]:
            print(az.summary(idata, var_names=["Ah", "Bh"]))

        az.plot_posterior(idata, var_names=["Ah", "Bh"])
        plt.savefig("posterior_plot.pdf")

        # Store data directly in object attributes
        self.last_trace = idata
        self.last_data = (dataX, dataS)
        # Save results to unique filenames
        self._save_results(idata, (dataX, dataS), method='xs')

    def run_inference_large_xs(self, samples=4000, tune=2000, cores=4) -> None:
        """
        Runs large-scale inference for the VAR model with metabolite data.

        Returns:
        None
        """
        if self.dataS is None:
            raise ValueError(
                "Metabolite data is missing. Please provide dataS.")

        if self.data is None:
            raise ValueError(
                "Abundance data is missing. Please provide dataX.")

        dataX = self.data
        dataS = self.dataS
        nX = dataX.shape[1]
        nS = dataS.shape[1]
        DA = nX * nX
        DA0 = 5
        DB = nS * nX
        DB0 = 4
        N = dataX.shape[0] - 2

        # Set priors if provided, else default to zero mean and unit variance
        x0_prior_mu = self.priors.get('intercepts', self.intercepts.flatten(
        )) if self.intercepts is not None else np.zeros(nX)
        A_prior_mu = self.priors.get(
            'coefficients',
            self.coefficients) if self.coefficients is not None else np.zeros(
            (nX,
             nX))
        noise_cov_prior = self.priors.get(
            'covariance_matrix',
            self.covariance_matrix) if self.covariance_matrix is not None else None

        with pm.Model() as var_model:
            tau0_A = (DA0 / (DA - DA0)) * 0.1 / np.sqrt(N)
            c2_A = pm.InverseGamma("c2_A", 2, 1)
            tau_A = pm.HalfCauchy("tau_A", beta=tau0_A)
            lam_A = pm.HalfCauchy("lam_A", beta=1, shape=(nX, nX))
            Ah = pm.Normal('Ah', mu=A_prior_mu, sigma=tau_A * lam_A * \
                           at.sqrt(c2_A / (c2_A + tau_A**2 * lam_A**2)), shape=(nX, nX))

            tau0_B = (DB0 / (DB - DB0)) * 0.1 / np.sqrt(N)
            c2_B = pm.InverseGamma("c2_B", 2, 1)
            tau_B = pm.HalfCauchy("tau_B", beta=tau0_B)
            lam_B = pm.HalfCauchy("lam_B", beta=1, shape=(nS, nX))
            Bh = pm.Normal('Bh', mu=0, sigma=tau_B * lam_B * \
                           at.sqrt(c2_B / (c2_B + tau_B**2 * lam_B**2)), shape=(nS, nX))

            if noise_cov_prior is not None:
                sigma = np.linalg.cholesky(noise_cov_prior)
            else:
                sigma = pm.TruncatedNormal(
                    'sigma', mu=0.1, sigma=0.1, lower=0, shape=(nX + nS))

            data = np.concatenate((dataX, dataS), axis=1)

            muX = pm.Deterministic('muX', pm.math.dot(Ah, dataX[:-1, :].T))
            muS = pm.math.dot(Bh, muX)
            muXs = muX[:, 1:]
            muSs = muS[:, :-1]
            mu = pm.math.concatenate((muXs, muSs), axis=0)

            likelihood = pm.Normal('likelihood', mu=mu.T,
                                   sigma=sigma, observed=data[2:, :])

        with var_model:
            trace = pm.sample(draws=samples, tune=tune, cores=cores)

        # print if `debug` is set to 'high' or 'low'
        if self.debug in ["high", "low"]:
            print(az.summary(trace, var_names=["Ah", "Bh"]))

        az.plot_posterior(trace, var_names=["Ah", "Bh"])
        plt.savefig("posterior_plot.pdf")

        # Store data directly in object attributes
        self.last_trace = trace
        self.last_data = (dataX, dataS)
        # Save results to unique filenames
        self._save_results(trace, (dataX, dataS), method='large_xs')

    def posterior_analysis(
            self,
            data_filename=None,
            netcdf_filename=None,
            A=None,
            B=None):
        """
        Performs posterior analysis and visualizes the results.

        Args:
        data_filename (str, optional): The filename for the .npz data file.
        netcdf_filename (str, optional): The filename for the NetCDF file with inference results.
        A (numpy.ndarray, optional): The true matrix A to compare against.
        B (numpy.ndarray, optional): The true matrix B to compare against.

        Returns:
        None
        """
        # Check if data is stored in attributes
        if self.last_data is not None:
            dataX, dataS = self.last_data[0], self.last_data[1] if len(
                self.last_data) > 1 else None
        else:
            # Load data from file if not stored in attributes
            data_filename = data_filename or self.last_data_filename
            try:
                with np.load(data_filename, allow_pickle=True) as xsdata:
                    dataX = xsdata['dataX']
                    dataS = xsdata['dataS'] if 'dataS' in xsdata else None
            except KeyError as e:
                print(f"Error loading data: {e}")
                dataS = None

        if dataS is not None:
            self.make_plot_stacked(dataX, dataS)

        # Check if inference data is stored in attributes
        if self.last_trace is not None:
            idata = self.last_trace
        else:
            # Load inference data from file if not stored in attributes
            netcdf_filename = netcdf_filename or self.last_netcdf_filename
            try:
                idata = az.from_netcdf(netcdf_filename)
                print(az.summary(idata, var_names=["Ah", "Bh"]))
            except FileNotFoundError as e:
                print(f"Error loading inference data: {e}")
                return

        true_values = [A, B] if A is not None and B is not None else None
        self.plot_heatmap(
            idata, matrices=["Ah", "Bh"], true_values=true_values)

    def make_plot_stacked(self, dataX, dataS):
        """
        Creates a stacked plot of abundance and metabolite data.

        Args:
        dataX (numpy.ndarray): The abundance data.
        dataS (numpy.ndarray): The metabolite data.

        Returns:
        None
        """
        dataX = dataX + 1.0

        nX = len(dataX[0])
        nS = len(dataS[0])
        nobs = dataS.shape[0]

        fig, axs = plt.subplots(2, 1, figsize=(10, 4))

        axs[0].stackplot(
            range(
                len(dataX)),
            *dataX.T,
            labels=[
                f"X{str(i)}" for i in range(nX)])
        axs[0].set_title("Abundance, log10 X")
        axs[0].set_ylabel("X")
        axs[0].set_xlim(0, nobs - 1)

        sns.heatmap(
            dataS.T,
            annot=False,
            cmap="YlGnBu",
            yticklabels=[f"S{str(i)}" for i in range(nS)],
            ax=axs[1],
            cbar=False,
        )
        axs[1].set_title("Metabolites, S")
        axs[1].set_ylabel("S")
        axs[1].set_xlabel("time (weeks)")
        axs[1].set_xlim(0, nobs)

        plt.tight_layout()
        plt.savefig("plot-data-XS-stacked.pdf")

    def plot_heatmap(self, idata, matrices=None, true_values=None):
        """
        Plots heatmaps of the inferred matrices.

        Args:
        idata (arviz.InferenceData): The inference data.
        matrices (list of str): List of keys for matrices in idata.posterior to plot.
        true_values (list of numpy.ndarray): List of true matrices to compare with.

        Returns:
        None
        """
        if matrices is None:
            matrices = ['Ah', 'Bh']  # default matrices to plot

        num_matrices = len(matrices)
        fig, axes = plt.subplots(
            1, num_matrices, figsize=(7 * num_matrices, 7))

        if num_matrices == 1:
            axes = [axes]  # ensure axes is a list for consistent indexing

        for idx, matrix_key in enumerate(matrices):
            if matrix_key not in idata.posterior:
                print(f"Key '{matrix_key}' not found in idata.posterior")
                continue

            matrix = idata.posterior[matrix_key].values
            matrix_sum = np.median(matrix, axis=(0, 1))

            sns.heatmap(matrix_sum, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(f'{matrix_key}hat')
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('X' if matrix_sum.shape[0]
                                 == matrix_sum.shape[1] else 'S')

            # Determine which values to annotate: true_values if provided, else
            # matrix_sum
            annotate_values = true_values[idx] if true_values is not None and len(
                true_values) > idx and true_values[idx] is not None else matrix_sum

            for i in range(matrix_sum.shape[0]):
                for j in range(matrix_sum.shape[1]):
                    text_color = 'white' if true_values is not None and len(
                        true_values) > idx and true_values[idx] is not None else 'black'
                    axes[idx].text(
                        j + 0.5,
                        i + 0.5,
                        f'{annotate_values[i, j]:.2f}',
                        ha='center',
                        va='center',
                        color=text_color)

        plt.tight_layout()
        plt.savefig('plot-posterior-heatmap.pdf', bbox_inches='tight')

    def _save_results(self, trace, data, method='default'):
        base_filename = f"model_posterior_{method}.nc"
        data_filename = f"data_{method}.npz"

        # Incremental versioning
        version = 1
        while os.path.exists(base_filename):
            base_filename = f"model_posterior_{method}_v{version}.nc"
            data_filename = f"data_{method}_v{version}.npz"
            version += 1

        # Save files as a fallback
        az.to_netcdf(trace, base_filename)
        dataX = data[0]
        dataS = data[1] if len(data) > 1 else None
        if dataS is None:
            np.savez(data_filename, dataX=dataX)
        else:
            np.savez(data_filename, dataX=dataX, dataS=dataS)

        # Store filenames as fallback
        self.last_data_filename = data_filename
        self.last_netcdf_filename = base_filename

        print(
            f"Results saved as:\nNetCDF file: {base_filename}\nData file: {data_filename}")

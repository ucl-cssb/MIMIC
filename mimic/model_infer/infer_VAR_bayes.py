import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at
import seaborn as sns


class infer_VAR:
    """
    infer_VAR class for performing inference on VAR models.

    This class provides methods for importing data, running inference on the VAR model, and performing posterior sampling and analysis.

    Args:
        data (numpy.ndarray): The data to perform inference on.

    Methods:
        import_data(file_path, index_col=None, parse_dates=False):
            Imports data from a .csv file.

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

    def __init__(self, data=None, coefficients=None, intercepts=None, covariance_matrix=None, dataS=None):
        self.data = self._validate_data(data)
        self.dataS = self._validate_data(dataS) if dataS is not None else None
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.covariance_matrix = covariance_matrix

    def _validate_data(self, data):
        if data is None:
            return None
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            try:
                return np.array(data)
            except Exception as e:
                raise TypeError(
                    f"Data could not be converted to a numpy array: {e}"
                ) from e
        else:
            raise TypeError(
                "Unsupported data type. Data must be a DataFrame, ndarray, list, or tuple.")

    def import_data(self, file_path, index_col=None, parse_dates=False, data_type='X') -> None:
        """
        Imports data from a .csv file.

        Args:
        file_path (str): The path to the .csv file.
        index_col (int, optional): Column to use as the row labels of the DataFrame.
        parse_dates (bool, optional): Parse dates as datetime.
        data_type (str): Specify whether the data is for 'X' (abundance) or 'S' (metabolite).

        Returns:
        None
        """
        try:
            data = pd.read_csv(file_path, index_col=index_col,
                               parse_dates=parse_dates)
            if data_type == 'X':
                self.data = self._validate_data(data)
            elif data_type == 'S':
                self.dataS = self._validate_data(data)
            else:
                raise ValueError("data_type must be either 'X' or 'S'.")
        except Exception as e:
            raise ValueError(
                f"Failed to import data from {file_path}: {e}") from e

    # sourcery skip: extract-method
    def run_inference(self, samples=2000, tune=1000, cores=2) -> None:
        """
        Runs the inference process for the VAR model.

        Returns:
        None
        """
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
        x0_prior_mu = self.intercepts.flatten() if self.intercepts is not None else np.zeros(
            dim)
        A_prior_mu = self.coefficients.values if self.coefficients is not None else np.zeros(
            (dim, dim))
        noise_cov_prior = self.covariance_matrix.values if self.covariance_matrix is not None else None
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
            print("x0:", x0.shape)
            # print("A:",A.shape)
            # print("data[:-1, :]:", data[:-1, :].shape)
            print("data[1:, :]:", data[1:, :].shape)
            x0_obs = data[0, :].copy().reshape(-1)
            print("x0:", x0_obs.shape)

            mu = x0[:, np.newaxis] + pm.math.dot(A, data[:-1, :].T)
            print("mu:", mu.T.shape)
            print("data:", data[1:, :].shape)

            # obs_chol = np.diag(np.full(dim,sigma))

            # *pm.Normal('likelihood_0', mu=x0, sigma=1.0, observed=x0_obs)
            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            # FIXME: #38 make these arguments specifiable in the
            # parameters.json file file
            trace = pm.sample(samples, tune=tune, cores=cores)

        # Plotting the posterior distributions
        # pm.plot_posterior(trace, var_names=['x0', 'A'])

        # Extracting the posterior means for initial values and coefficients
        # posterior_means = {
        #    'x0': np.mean(trace['x0'], axis=0),
        #    'A': np.mean(trace['A'], axis=0)
        #

        # print("Posterior Means:")
        # print(posterior_means)

        print(az.summary(trace, var_names=["x0", "A"]))

        az.plot_posterior(trace, var_names=["x0", "A"])
        plt.savefig("posterior_plot.pdf")
        az.to_netcdf(trace, 'model_posterior.nc')
        np.savez("data.npz", dataX=data)

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
        x0_prior_mu = self.intercepts.flatten(
        ) if self.intercepts is not None else np.zeros(ndim)
        A_prior_mu = self.coefficients.values if self.coefficients is not None else np.zeros(
            (ndim, ndim))
        noise_cov_prior = self.covariance_matrix.values if self.covariance_matrix is not None else None

        # create and fit PyMC model
        with pm.Model() as var_model:
            # Standard LKJ priors Priors for x0 and sigma
            # x0 = pm.Normal('x0', mu=initial_values_true, sigma=0.01, shape=(ndim,1))
            # A = pm.Normal('A', mu=0, sigma=1, shape=(ndim, ndim))

            # Priors for coefficients with LKJ prior
            # noise_chol, _, _ = pm.LKJCholeskyCov("noise_chol", eta=1.0, n=dim, sd_dist=pm.HalfNormal.dist(sigma=1.0) )

            # VAR(1) process likelihood
            # mu = x0 + pm.math.dot(A, data[:-1, :].T)
            # likelihood = pm.MvNormal('likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

            # Priors for coefficients with horseshoe -> sparse VAR
            noise_stddev = pm.HalfNormal("noise_stddev", 25)
            # [0,0]*ndim might be better
            x0 = pm.Normal('x0', mu=x0_prior_mu,
                           sigma=0.001, shape=(ndim,))

            # Standard horse shoe
            # Prior on error SD
            # sigma = pm.HalfNormal("sigma", 25)
            # Global shrinkage prior
            # tau = pm.HalfStudentT("tau", 2, D0 / (D - D0) * sigma / np.sqrt(N))
            # Local shrinkage prior
            # lam = pm.HalfStudentT("lam", 5, shape=(ndim, ndim) )
            # c2 = pm.InverseGamma("c2", 2, 8)
            # z = pm.Normal("z", 0.0, 1.0, shape=(ndim, ndim) )
            # Shrunken coefficients
            # A = pm.Normal('A', mu=0, sigma = z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), shape=(ndim, ndim) )

            # Regularised horse shoe
            tau0 = (D0 / (D - D0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 8)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy("lam", beta=1, shape=(ndim, ndim))
            A = pm.Normal('A', mu=A_prior_mu, sigma=tau * lam * at.sqrt(c2 /
                          (c2 + tau**2 * lam**2)), shape=(ndim, ndim))

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
            trace = pm.sample(samples, tune=tune, cores=cores)

        print(az.summary(trace, var_names=["A"]))

        az.plot_posterior(trace, var_names=[
            "A"])
        plt.savefig("plot-posterior.pdf")
        az.to_netcdf(trace, 'model_posterior.nc')
        np.savez("data.npz", dataX=data)

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
        x0_prior_mu = self.intercepts.flatten(
        ) if self.intercepts is not None else np.zeros(nX)
        A_prior_mu = self.coefficients.values if self.coefficients is not None else np.zeros(
            (nX, nX))
        noise_cov_prior = self.covariance_matrix.values if self.covariance_matrix is not None else None

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
            idata = pm.sample(samples, tune=tune, cores=cores)

        print(az.summary(idata, var_names=["Ah", "Bh"]))
        az.to_netcdf(idata, 'model_posterior.nc')
        np.savez("data.npz", dataX=dataX, dataS=dataS)

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
        x0_prior_mu = self.intercepts.flatten(
        ) if self.intercepts is not None else np.zeros(nX)
        A_prior_mu = self.coefficients.values if self.coefficients is not None else np.zeros(
            (nX, nX))
        noise_cov_prior = self.covariance_matrix.values if self.covariance_matrix is not None else None

        with pm.Model() as var_model:
            tau0_A = (DA0 / (DA - DA0)) * 0.1 / np.sqrt(N)
            c2_A = pm.InverseGamma("c2_A", 2, 1)
            tau_A = pm.HalfCauchy("tau_A", beta=tau0_A)
            lam_A = pm.HalfCauchy("lam_A", beta=1, shape=(nX, nX))
            Ah = pm.Normal('Ah', mu=A_prior_mu, sigma=tau_A * lam_A *
                           at.sqrt(c2_A / (c2_A + tau_A**2 * lam_A**2)), shape=(nX, nX))

            tau0_B = (DB0 / (DB - DB0)) * 0.1 / np.sqrt(N)
            c2_B = pm.InverseGamma("c2_B", 2, 1)
            tau_B = pm.HalfCauchy("tau_B", beta=tau0_B)
            lam_B = pm.HalfCauchy("lam_B", beta=1, shape=(nS, nX))
            Bh = pm.Normal('Bh', mu=0, sigma=tau_B * lam_B *
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
            trace = pm.sample(samples, tune=tune, cores=cores)

        print(az.summary(trace, var_names=["Ah", "Bh"]))
        az.to_netcdf(trace, 'model_posterior.nc')
        np.savez("data.npz", dataX=dataX, dataS=dataS)

    def posterior_analysis(self, A=None, B=None):
        """
        Performs posterior analysis and visualizes the results.

        Args:
        simulated (bool): Indicates whether the data is simulated or not.

        Returns:
        None
        """
        with np.load("data.npz", allow_pickle=True) as xsdata:
            dataX = xsdata['dataX']
            dataS = xsdata['dataS']

        self.make_plot_stacked(dataX, dataS)

        idata = az.from_netcdf('model_posterior.nc')

        print(az.summary(idata, var_names=["Ah", "Bh"]))

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
            range(len(dataX)), *dataX.T, labels=[f"X{str(i)}" for i in range(nX)]
        )
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
            axes[idx].set_ylabel('X')
            axes[idx].set_xlabel('X' if matrix_sum.shape[0]
                                 == matrix_sum.shape[1] else 'S')

            if true_values is not None and len(true_values) > idx and true_values[idx] is not None:
                true_matrix = true_values[idx]
                for i in range(matrix_sum.shape[0]):
                    for j in range(matrix_sum.shape[1]):
                        axes[idx].text(
                            j + 0.5, i + 0.5, f'{true_matrix[i, j]:.2f}', ha='center', va='center', color='white')

        plt.tight_layout()
        plt.savefig('plot-posterior-heatmap.pdf', bbox_inches='tight')

    # def plot_heatmap(self, idata, A=None, B=None):
    #     """
    #     Plots heatmaps of the inferred A and B matrices.

    #     Args:
    #     idata (arviz.InferenceData): The inference data.
    #     A (numpy.ndarray): The true A matrix.
    #     B (numpy.ndarray): The true B matrix.

    #     Returns:
    #     None
    #     """
    #     matrix1 = idata.posterior['Ah'].values
    #     matrix2 = idata.posterior['Bh'].values

    #     matrix1_sum = np.median(matrix1, axis=(0, 1))
    #     matrix2_sum = np.median(matrix2, axis=(0, 1))

    #     fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={
    #                            'width_ratios': [1, 1.2]})

    #     sns.heatmap(matrix1_sum, ax=ax[0], cmap='viridis')
    #     ax[0].set_title('Ahat')
    #     ax[0].set_ylabel('X')
    #     ax[0].set_xlabel('X')
    #     if A is not None:
    #         for i in range(matrix1_sum.shape[0]):
    #             for j in range(matrix1_sum.shape[1]):
    #                 ax[0].text(
    #                     j + 0.5, i + 0.5, f'{A[i, j]:.2f}', ha='center', va='center', color='white')

    #     matrix2_sum = matrix2_sum.T

    #     sns.heatmap(matrix2_sum, ax=ax[1], cmap='viridis')
    #     ax[1].set_title('Bhat')
    #     ax[1].set_xlabel('S')

    #     if B is not None:
    #         BT = B.T
    #         for i in range(matrix2_sum.shape[0]):
    #             for j in range(matrix2_sum.shape[1]):
    #                 ax[1].text(
    #                     j + 0.5, i + 0.5, f'{BT[i, j]:.2f}', ha='center', va='center', color='white')

    #     plt.savefig('plot-posterior-heatmap.pdf', bbox_inches='tight')


# Path: mimic/model_infer/infer_VAR.py

# Example usage:
# infer = infer_VAR(dataX, dataS)
# infer.run_inference_xs()
# infer.posterior_analysis(simulated=True)

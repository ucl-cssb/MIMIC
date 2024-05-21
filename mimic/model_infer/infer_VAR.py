import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at


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

    Returns:
        None
    """

    def __init__(self, data, coefficients=None, intercepts=None, covariance_matrix=None):
        self.data = self._validate_data(data)
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.covariance_matrix = covariance_matrix

    def _validate_data(self, data):
        if isinstance(data, pd.DataFrame):
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

    def import_data(self, file_path, index_col=None, parse_dates=False) -> None:
        """
        Imports data from a .csv file.

        Args:
        file_path (str): The path to the .csv file.
        index_col (int, optional): Column to use as the row labels of the DataFrame.
        parse_dates (bool, optional): Parse dates as datetime.

        Returns:
        None
        """
        try:
            data = pd.read_csv(file_path, index_col=index_col,
                               parse_dates=parse_dates)
            self.data = self._validate_data(data)
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
        noise_cov_prior = self.covariance_matrix.values if self.covariance_matrix is not None else np.eye(
            dim)
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

    def run_inference_large(self, samples=4000, tune=2000, cores=4) -> None:
        """
        Run large-scale inference for VAR model.

        This function generates VAR model data, fits a Bayesian VAR model using PyMC3,
        and performs posterior sampling and analysis.

        Returns:
            None
        """
        data = self.data
        # Check the dimensions of the data
        ndim = data.shape[1]

        # Params for shrinkage
        D = ndim**2
        D0 = 2
        N = data.shape[0]

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
            x0 = pm.Normal('x0', mu=0,
                           sigma=0.001, shape=(ndim, 1))

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
            A = pm.Normal('A', mu=0, sigma=tau * lam * at.sqrt(c2 /
                          (c2 + tau**2 * lam**2)), shape=(ndim, ndim))

            # Priors for coefficients with LKJ prior
            noise_chol, _, _ = pm.LKJCholeskyCov(
                "noise_chol", eta=1.0, n=ndim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

            # VAR(1) process likelihood
            mu = x0 + pm.math.dot(A, data[:-1, :].T)
            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            trace = pm.sample(samples, tune=tune, cores=cores)

        print(az.summary(trace, var_names=["A"]))

        az.plot_posterior(trace, var_names=[
            "A"])
        plt.savefig("plot-posterior.pdf")

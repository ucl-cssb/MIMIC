import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as at
from VARsim import VARSimulator
from utils import read_parameters


class VARInfer:
    def __init__(self, data):
        self.data = data  # data to do inference on

    def run_inference(self):  # sourcery skip: extract-method
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
        with pm.Model() as var_model:
            # Priors for x0 and sigma
            # QUESTION: should the sigma be the noise_stddev from the parameters.json file?
            x0 = pm.Normal('x0', mu=0, sigma=1, shape=(dim, 1))
            A = pm.Normal('A', mu=0, sigma=1, shape=(dim, dim))

            # Priors for coefficients with LKJ prior
            # packed_L = pm.LKJCholeskyCov('packed_L', n=dim, eta=2.0, sd_dist=pm.HalfCauchy.dist(2.5))
            # L = pm.expand_packed_triangular(dim, packed_L)
            # coefficients = pm.MvNormal('coefficients', mu=0, chol=L, shape=(dim, dim))

            noise_chol, _, _ = pm.LKJCholeskyCov(
                "noise_chol", eta=1.0, n=dim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

            # VAR(1) process likelihood
            print("x0:", x0.shape)
            # print("A:",A.shape)
            # print("data[:-1, :]:", data[:-1, :].shape)
            print("data[1:, :]:", data[1:, :].shape)
            x0_obs = data[0, :].copy().reshape(2, 1)
            print("x0:", x0_obs.shape)

            mu = x0 + pm.math.dot(A, data[:-1, :].T)
            print("mu:", mu.T.shape)
            print("data:", data[1:, :].shape)

            # obs_chol = np.diag(np.full(dim,sigma))

            # *pm.Normal('likelihood_0', mu=x0, sigma=1.0, observed=x0_obs)
            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            # FIXME: make these arguments specifiable in the parameters.json file file
            trace = pm.sample(2000, tune=1000, cores=2)

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

    def run_inference_large(self):
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
            # HACK: mu = 0 was addded by me. It might be wrong and mu = [0,0]*ndim might be better
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
            A = pm.Normal('A', mu=0, sigma=tau * lam*at.sqrt((c2 /
                          (c2 + tau**2 * lam**2)), shape=(ndim, ndim)))

            # Priors for coefficients with LKJ prior
            noise_chol, _, _ = pm.LKJCholeskyCov(
                "noise_chol", eta=1.0, n=ndim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

            # VAR(1) process likelihood
            mu = x0 + pm.math.dot(A, data[:-1, :].T)
            likelihood = pm.MvNormal(
                'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Sampling from the posterior
        with var_model:
            trace = pm.sample(2000, tune=1000, cores=4)

        print(az.summary(trace, var_names=["A"]))

        az.plot_posterior(trace, var_names=[
            "A"])
        plt.savefig("plot-posterior.pdf")


if __name__ == '__main__':
    # Import parameters from JSON file

    parameters = read_parameters('parameters.json')
    simulator = VARSimulator(**parameters)
    simulator.run("VARsim")

    infer = VARInfer(simulator.data)
    infer.run_inference()

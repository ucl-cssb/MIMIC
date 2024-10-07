import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as at
import pickle
import cloudpickle

from mimic.utilities import *
from mimic.model_simulate.sim_gLV import *


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


def get_data(input_data):
    # Read the CSV file
    d = pd.read_csv(input_data)

    # Calculate the mean of columns 2 to 5 (index 1 to 4) for each row
    # Take only time course up to t=400
    X1_bar = d.iloc[1:21, 1:5].mean(axis=1)

    # Calculate the mean of columns 6 to 9 (index 5 to 8) for each row
    X2_bar = d.iloc[1:21, 5:9].mean(axis=1)

    # Combine the first column with the calculated means
    obs = pd.DataFrame({
        'time': d.iloc[1:21, 0],
        'X1_bar': X1_bar,
        'X2_bar': X2_bar
    })

    # Replace negative values with 0
    obs[obs < 0] = 0

    return obs


def plot_growth_curves(data):
    plt.figure(figsize=(10, 6))

    # Plotting X1_bar
    plt.plot(data['time'], data['X1_bar'], label='X1_bar')

    # Plotting X2_bar
    plt.plot(data['time'], data['X2_bar'], label='X2_bar')

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Growth Curves of X1_bar and X2_bar')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()


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

    def __init__(
            self,
            X=None,
            F=None,
            mu=None,
            M=None,
            prior_mu_mean = None,
            prior_mu_sigma = None,
            prior_Mii_mean=None,
            prior_Mii_sigma = None,
            prior_Mij_sigma = None,
            prior_eps_mean = None,
            prior_eps_sigma = None,
            draws = None,
            tune = None,
            chains = None,
            cores = None,
            M_h=None,
            DA=None,
            DA0=None,
            N=None,
            noise_stddev=None,
            epsilon=None,
            sim_glv=None):

        # self.data = data  # data to do inference on
        self.X = X
        self.F = F
        self.mu = mu
        self.M = M
        self.prior_mu_mean = prior_mu_mean
        self.prior_mu_sigma = prior_mu_sigma
        self.prior_Mii_mean = prior_Mii_mean
        self.prior_Mii_sigma = prior_Mii_sigma
        self.prior_Mij_sigma = prior_Mij_sigma
        self.prior_eps_mean = prior_eps_mean
        self.prior_eps_sigma = prior_eps_sigma
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.M_h = M_h
        self.DA = DA
        self.DA0 = DA0 if DA0 is not None else (
            self.calculate_DA0(F.shape[1]) if F is not None else None)
        self.N = N
        self.noise_stddev = noise_stddev
        self.epsilon = epsilon
        self.sim_glv = sim_glv

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

    def run_bayes_gLV(self) -> None:
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
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal('mu_hat', mu=prior_mu_mean, sigma=prior_mu_sigma, lower=0, shape=( 1, num_species))

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.Normal('M_ii_hat_p',mu=prior_Mii_mean, sigma=prior_Mii_sigma, shape=(num_species,))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ij is unconstrained
            M_ij_hat = pm.HalfNormal('M_ij_hat', sigma=prior_Mij_sigma, shape=(num_species, num_species - 1))  # different shape for off-diagonal

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
            # model_mean = pm.math.dot(X, pm.math.concatenate([M_hat_vals, mu_hat], axis=0))
            model_mean = pm.math.dot(X, pm.math.concatenate([M_hat, mu_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # For debugging:

            # As tensor objects are symbolic, if needed print using .eval()
            # eg
            # print(f"mu_hat: {mu_hat.eval()}")

            # initial_values = bayes_model.initial_point()
            # print(f"Initial parameter values: {initial_values}")

            # Posterior distribution
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)

        # Assemble posterior values for mu and M for plotting and assessment
        mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        # Plot and save posterior results
        self.plot_posterior(idata, mu_hat_np, M_hat_np)

        return idata

    def run_bayes_gLV_shrinkage(self) -> None:
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
        print(f"DA: {DA}, DA0: {DA0}, N: {N}, noise_stddev: {noise_stddev}")

        num_species = F.shape[1]

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  #
            # A separate sigma for each response
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal('mu_hat', mu=prior_mu_mean, sigma=prior_mu_sigma, lower=0,
                                        shape=(1, num_species))

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.Normal('M_ii_hat_p', mu=prior_Mii_mean, sigma=prior_Mii_sigma, shape=(num_species,))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)


            # M_ii_hat = pm.TruncatedNormal('M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(num_species,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy(
                "lam", beta=1, shape=(
                    num_species, num_species - 1))
            # M_ij_hat = pm.Normal('M_ij_hat', mu=M_prior, sigma=tau * lam *
            # at.sqrt(c2 / (c2 + tau ** 2 * lam ** 2)), shape=(num_species,
            # num_species-1))
            M_ij_hat = pm.HalfNormal('M_ij_hat', sigma=prior_Mij_sigma, shape=(num_species, num_species - 1))  # different shape for off-diagonal

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
            model_mean = pm.math.dot(X, pm.math.concatenate([M_hat, mu_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # For debugging:

            # As tensor objects are symbolic, if needed print using .eval()
            # eg
            # print(f"mu_hat: {mu_hat.eval()}")

            # initial_values = bayes_model.initial_point()
            # print(f"Initial parameter values: {initial_values}")

            # Posterior distribution
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)

            # Assemble posterior values for mu and M for plotting and
            # assessment
        mu_hat_np = idata.posterior['mu_hat'].mean( dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        # Plot and save posterior results
        self.plot_posterior(idata, mu_hat_np, M_hat_np)

        return idata

    def run_bayes_gLV_shrinkage_pert(self) -> None:
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
        print(f"DA: {DA}, DA0: {DA0}, N: {N}, noise_stddev: {noise_stddev}")

        num_species = F.shape[1]

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            # sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  #
            # A separate sigma for each response
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # Define mu as prior
            mu_hat = pm.TruncatedNormal('mu_hat', mu=prior_mu_mean, sigma=prior_mu_sigma, lower=0,
                                        shape=(1, num_species))

            epsilon_hat = pm.Normal('epsilon_hat', mu=prior_eps_mean, sigma=prior_eps_sigma, shape=(1, num_species))


            # Set constraints for horseshoe prior
            # M_ij is is unconstrained but placed under horseshoe prior to
            # apply to sigma for M_ij

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.Normal('M_ii_hat_p', mu=prior_Mii_mean, sigma=prior_Mii_sigma, shape=(num_species,))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ii_hat = pm.TruncatedNormal('M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(num_species,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy(
                "lam", beta=1, shape=(
                    num_species, num_species - 1))
            # M_ij_hat = pm.Normal('M_ij_hat', mu=M_prior, sigma=tau * lam *
            # at.sqrt(c2 / (c2 + tau ** 2 * lam ** 2)), shape=(num_species,
            # num_species-1))
            M_ij_hat = pm.HalfNormal('M_ij_hat', sigma=prior_Mij_sigma, shape=(num_species, num_species - 1))

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
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)

            # Assemble posterior values for mu and M for plotting and
            # assessment
        mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        # Plot and save posterior results
        self.plot_posterior_pert(idata, mu_hat_np, M_hat_np, epsilon)

        return idata

    def plot_posterior(self, idata, mu_hat_np, M_hat_np):
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

    def plot_posterior_pert(self, idata, mu_hat_np, M_hat_np, epsilon):
        az.plot_posterior(idata, var_names=["mu_hat"], ref_val=mu_hat_np.tolist())
        plt.savefig("plot-posterior-mu.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(idata, var_names=["M_ii_hat"], ref_val=np.diag(M_hat_np).tolist())
        plt.savefig("plot-posterior-Mii.pdf")
        plt.show()
        plt.close()

        mask = ~np.eye(M_hat_np.shape[0], dtype=bool)
        M_ij = M_hat_np[mask]
        az.plot_posterior(idata, var_names=["M_ij_hat"], ref_val=M_ij.flatten().tolist())
        plt.savefig("plot-posterior-Mij.pdf")
        plt.show()
        plt.close()

        az.plot_posterior(idata, var_names=["epsilon_hat"], ref_val=epsilon.flatten().tolist())
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


def param_data_compare(idata, F, mu, M, times, yobs, init_species_start, sim_gLV_class):
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


def curve_compare(idata, F, mu, M, times, yobs, init_species_start, sim_gLV_class):
    # Compare model parameters to the data
    num_species = F.shape[1]
    # init_species = 10 * np.ones(num_species)
    init_species = 0.01 * np.ones(num_species)

    print(idata.posterior["M_hat"].values.shape)

    print(idata.posterior["mu_hat"].values.shape)

    # # get median posterior values
    M_h = np.median(idata.posterior["M_hat"].values, axis=(0, 1))

    mu_h = np.median(idata.posterior["mu_hat"].values, axis=(0, 1))
    mu_h = mu_h.flatten()

    # mu_h = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
    # M_h= idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

    predictor = sim_gLV(num_species=num_species,
                        M=M_h,
                        mu=mu_h
                        )
    yobs_h, _, _, _, _ = predictor.simulate(
        times=times, init_species=init_species)

    plot_fit_gLV(yobs, yobs_h, times)


def generate_5_species_data(sim_gLV_class):
    # In this example n >> p and it is basically same as standard regression
    # We have to be careful as most of these gLV models are very weakly
    # identifiable

    set_all_seeds(1234)

    # SETUP MODEL
    # establish size of model
    num_species = 5

    # construct interaction matrix
    # TODO do this programmatically
    M = np.zeros((num_species, num_species))
    np.fill_diagonal(M, [-0.05, -0.1, -0.15, -0.01, -0.2])
    M[0, 2] = -0.025
    M[1, 3] = 0.05
    M[4, 0] = 0.02

    # construct growth rates matrix
    mu = np.random.lognormal(0.01, 0.5, num_species)
    print(mu.shape)

    # instantiate simulator
    simulator = sim_gLV(num_species=num_species,
                        M=M,
                        mu=mu)
    simulator.print_parameters()

    # PRODUCE SIMULATED RESULTS
    # initial conditions
    init_species = 10 * np.ones(num_species)

    times = np.arange(0, 5, 0.1)
    yobs, init_species, mu, M, _ = simulator.simulate(
        times=times, init_species=init_species)

    # add some gaussian noise
    yobs = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)

    # plot simulation
    plot_gLV(yobs, times)

    return yobs, times, mu, M


def pert_fn(t):
    if 2.0 <= t < 2.2 or 3.0 <= t < 3.2 or 4.0 <= t < 4.2:
        return np.array([1])
    else:
        return np.array([0])


def param_data_compare_pert( idata, F, mu, M, epsilon, num_perturbations, times, yobs, init_species_start, sim_gLV_class):
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

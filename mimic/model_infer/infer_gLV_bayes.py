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

    def __init__(
            self,
            X=None,
            F=None,
            mu=None,
            M=None,
            M_h=None,
            DA=None,
            DA0=None,
            N=None,
            noise_stddev=None,
            epsilon=None):
        # self.data = data  # data to do inference on
        self.X = X
        self.F = F
        self.mu = mu
        self.M = M
        self.M_h = M_h
        self.DA = DA
        self.DA0 = DA0 if DA0 is not None else (self.calculate_DA0(F.shape[1]) if F is not None else None)
        self.N = N
        self.noise_stddev = noise_stddev
        self.epsilon = epsilon

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


        """

        #if self.X is None or self.F is None or self.mu is None or self.M is None:
        #    raise ValueError("X, F, mu, and M must all be provided.")

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M

        num_species = F.shape[1]

        # Print shapes for debugging
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            #sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  # A separate sigma for each response
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # If mu_value is provided, use it as a fixed value; otherwise, define it as a prior
            if mu is not None:
                mu = mu
            else:
                mu = pm.Uniform('mu', lower=0.0, upper=0.05)

            # mu_hat = pm.HalfNormal('mu_hat', sigma=1, shape=(1, 5))
            mu_hat = pm.TruncatedNormal('mu_hat', mu=mu, sigma=0.5, lower=0, shape=(1, num_species))

            #M_hat = pm.Normal('M_hat', mu=0, sigma=0.1, shape=(num_species, num_species))  #
            # tighter prior for the slopes

            # M_ii is constrained to be negative
            M_ii_hat_p = pm.HalfNormal('M_ii_hat_p', sigma=0.1, shape=(num_species,))
            M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

            # M_ij is unconstrained
            M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=0.1, shape=(num_species, num_species - 1))  # different shape for off-diagonal

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((num_species, num_species))
            M_hat_vals = at.set_subtensor(
                M_hat_vals[at.arange(num_species), at.arange(num_species)], M_ii_hat)  # set diagonal

            # Create the index arrays for setting off-diagonal values
            row_indices = np.arange(num_species)[:, None]
            col_indices = np.delete(np.arange(num_species), -1)

            M_hat_vals = at.set_subtensor(M_hat_vals[row_indices, col_indices], M_ij_hat)  # set off-diagonal

            # Save the combined matrix as a deterministic variable
            #M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat_vals, mu_hat], axis=0))

            # Priors for mu and M
            #mu = pm.Normal('mu', mu=0, sigma=1, shape=(1, num_species))

            #M = pm.Normal('M', mu=0, sigma=0.1, shape=(num_species, num_species))


            # Expected value of outcome (linear model) using mu and M
            #model_mean = pm.math.dot(
            #    X, pm.math.concatenate([M, mu], axis=0))

            print(f"model mean before: {model_mean.eval()}")

            log_model_mean = at.log(model_mean)

            print(f"log model mean after: {log_model_mean.eval()}")



            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Lognormal('Y_obs', mu=log_model_mean, sigma=sigma, observed=F)

            print(f"Y_obs logged: {Y_obs.eval()}")



            # Posterior distribution
            idata = pm.sample(500, tune=500, chains=2, cores=1)


        #print(f"idata type inside function: {type(idata)}")

        #mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        #M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        #print(f"mu_hat_np shape: {mu_hat_np.shape}")
        #print(f"M_hat_np shape: {M_hat_np.shape}")

        # Plot and save posterior results
        self.plot_posterior_a(idata, mu_hat_np, M_hat_np)

        print(f"idata type after posterior extraction: {type(idata)}")

        return idata



    def run_bayes_gLV_test(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model

        Returns:
            idata: The posterior inference data


        """

        #if self.X is None or self.F is None or self.mu is None or self.M is None:
        #    raise ValueError("X, F, mu, and M must all be provided.")

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M

        num_species = F.shape[1]

        # Print shapes for debugging
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            #sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  # A separate sigma for each response
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # Use provided mu as fixed value or define as prior
            if mu is not None:
                mu_hat = pm.TruncatedNormal('mu_hat', mu=mu, sigma=0.5, lower=0, shape=(1, num_species))
                print(f"Used manually determined mu")
            else:
                mu_hat = pm.TruncatedNormal('mu_hat', mu=pm.Uniform('mu', lower=0.0, upper=0.05), sigma=0.5, lower=0,
                                            shape=(1, num_species))
                print(f"Used automatically determined mu")

            # Use provided M as fixed values or define as priors
            if M is not None:
                    # Initialize an empty tensor of the same shape as prior
                    M_hat_vals = at.zeros((num_species, num_species))
                    print(f"Used manually determined M")

                    diagonal_var_names = []
                    off_diagonal_var_names = []

                    for i in range(num_species):
                        for j in range(num_species):
                            if i == j:
                                # Diagonal elements
                                M_ii_hat_prior = M[i, j]  # Original value from matrix A
                                M_ii_hat = pm.Deterministic(f'M_{i}_{j}_hat',
                                                            -pm.Normal(f'M_{i}_{j}_hat_prior', mu=M_ii_hat_prior, sigma=0.1))

                                # Insert into the tensor
                                M_hat_vals = at.set_subtensor(M_hat_vals[i, j], M_ii_hat)
                                diagonal_var_names.append(f'M_{i}_{j}_hat')
                            else:
                                # Off-diagonal elements
                                M_ij_hat_prior = M[i, j]  # Original value from matrix A
                                M_ij_hat = pm.Normal(f'M_{i}_{j}_hat_prior', mu=M_ij_hat_prior, sigma=0.1)

                                # Insert into the tensor
                                M_hat_vals = at.set_subtensor(M_hat_vals[i, j], M_ij_hat)
                                off_diagonal_var_names.append(f'M_{i}_{j}_hat_prior')




            else:
                # M_ii is constrained to be negative
                M_ii_hat_p = pm.HalfNormal('M_ii_hat_p', sigma=0.1, shape=(num_species,))
                M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

                # M_ij is unconstrained
                M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=0.1,
                                     shape=(num_species, num_species - 1))  # different shape for off-diagonal

                # Combine values
                # start with an all-zero matrix of the correct shape
                M_hat_vals = at.zeros((num_species, num_species))
                M_hat_vals = at.set_subtensor(
                    M_hat_vals[at.arange(num_species), at.arange(num_species)], M_ii_hat)  # set diagonal

                # Create the index arrays for setting off-diagonal values
                row_indices = np.arange(num_species)[:, None]
                col_indices = np.delete(np.arange(num_species), -1)

                M_hat_vals = at.set_subtensor(M_hat_vals[row_indices, col_indices], M_ij_hat)  # set off-diagonal

                # Generate variable names
                diagonal_var_names = [f'M_ii_hat']
                off_diagonal_var_names = [f'M_ij_hat']

                print(f"Used automatically determined M")


            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            print(f"mu_hat: {mu_hat.eval()}")
            print(f"M_hat_vals: {M_hat_vals.eval()}")
            print(f"M_hat: {M_hat.eval()}")

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat, mu_hat], axis=0))

            # Priors for mu and M
            #mu = pm.Normal('mu', mu=0, sigma=1, shape=(1, num_species))

            #M = pm.Normal('M', mu=0, sigma=0.1, shape=(num_species, num_species))


            # Expected value of outcome (linear model) using mu and M
            #model_mean = pm.math.dot(
            #    X, pm.math.concatenate([M, mu], axis=0))

            print(f"model mean before: {model_mean.eval()}")

            ##log_model_mean = at.log(model_mean)

            ##print(f"log model mean after: {log_model_mean.eval()}")



            # Likelihood (sampling distribution) of observations
            ##Y_obs = pm.Lognormal('Y_obs', mu=log_model_mean, sigma=sigma, observed=F)
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            ##print(f"Y_obs logged: {Y_obs.eval()}")
            print(f"Y_obs: {Y_obs.eval()}")

            # Print initial parameter values and logp values for debugging
            initial_values = bayes_model.initial_point()
            ##initial_logp = bayes_model.logp(initial_values)

            print(f"Initial parameter values: {initial_values}")
            ##print(f"Initial logp values: {initial_logp}")

            # Posterior distribution
            idata = pm.sample(1000, tune=1000, chains=4, cores=4)


        #print(f"idata type inside function: {type(idata)}")

        mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        print(f"mu_hat_np shape: {mu_hat_np.shape}")
        print(f"M_hat_np shape: {M_hat_np.shape}")

        var_names = diagonal_var_names + off_diagonal_var_names

        # Plot and save posterior results
        self.plot_posterior_a(idata, mu_hat_np, M_hat_np, diagonal_var_names, off_diagonal_var_names)

        print(f"idata type after posterior extraction: {type(idata)}")

        return idata, var_names



    def run_bayes_gLV_test_log(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model

        Returns:
            idata: The posterior inference data


        """

        #if self.X is None or self.F is None or self.mu is None or self.M is None:
        #    raise ValueError("X, F, mu, and M must all be provided.")

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M

        num_species = F.shape[1]

        # ensure F values are positive and non-zero
        F = np.where(F <= 0, 1e-6, F)

        # Print shapes for debugging
        print(f"X shape: {X.shape}")
        print(f"F shape: {F.shape}")
        print(f"Number of species: {num_species}")

        bayes_model = pm.Model()
        with bayes_model:
            # Priors for unknown model parameters
            #sigma = pm.HalfNormal('sigma', sigma=1, shape=(num_species,))  # A separate sigma for each response
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # Use provided mu as fixed value or define as prior
            if mu is not None:
                mu_hat = pm.TruncatedNormal('mu_hat', mu=mu, sigma=0.5, lower=0, shape=(1, num_species))
                print(f"Used manually determined mu")
            else:
                mu_hat = pm.TruncatedNormal('mu_hat', mu=pm.Uniform('mu', lower=0.0, upper=0.05), sigma=0.5, lower=0,
                                            shape=(1, num_species))
                print(f"Used automatically determined mu")

            # Use provided M as fixed values or define as priors
            if M is not None:
                    # Initialize an empty tensor of the same shape as prior
                    M_hat_vals = at.zeros((num_species, num_species))
                    print(f"Used manually determined M")

                    diagonal_var_names = []
                    off_diagonal_var_names = []

                    for i in range(num_species):
                        for j in range(num_species):
                            if i == j:
                                # Diagonal elements
                                M_ii_hat_prior = M[i, j]  # Original value from matrix A
                                M_ii_hat = pm.Deterministic(f'M_{i}_{j}_hat',
                                                            -pm.Normal(f'M_{i}_{j}_hat_prior', mu=M_ii_hat_prior, sigma=0.1))

                                # Insert into the tensor
                                M_hat_vals = at.set_subtensor(M_hat_vals[i, j], M_ii_hat)
                                diagonal_var_names.append(f'M_{i}_{j}_hat')
                            else:
                                # Off-diagonal elements
                                M_ij_hat_prior = M[i, j]  # Original value from matrix A
                                M_ij_hat = pm.Normal(f'M_{i}_{j}_hat_prior', mu=M_ij_hat_prior, sigma=0.1)

                                # Insert into the tensor
                                M_hat_vals = at.set_subtensor(M_hat_vals[i, j], M_ij_hat)
                                off_diagonal_var_names.append(f'M_{i}_{j}_hat_prior')




            else:
                # M_ii is constrained to be negative
                M_ii_hat_p = pm.HalfNormal('M_ii_hat_p', sigma=0.1, shape=(num_species,))
                M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)

                # M_ij is unconstrained
                M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=0.1,
                                     shape=(num_species, num_species - 1))  # different shape for off-diagonal

                # Combine values
                # start with an all-zero matrix of the correct shape
                M_hat_vals = at.zeros((num_species, num_species))
                M_hat_vals = at.set_subtensor(
                    M_hat_vals[at.arange(num_species), at.arange(num_species)], M_ii_hat)  # set diagonal

                # Create the index arrays for setting off-diagonal values
                row_indices = np.arange(num_species)[:, None]
                col_indices = np.delete(np.arange(num_species), -1)

                M_hat_vals = at.set_subtensor(M_hat_vals[row_indices, col_indices], M_ij_hat)  # set off-diagonal

                # Generate variable names
                diagonal_var_names = [f'M_ii_hat']
                off_diagonal_var_names = [f'M_ij_hat']

                print(f"Used automatically determined M")


            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            print(f"mu_hat: {mu_hat.eval()}")
            print(f"M_hat_vals: {M_hat_vals.eval()}")
            print(f"M_hat: {M_hat.eval()}")

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat, mu_hat], axis=0))



            print(f"model mean before: {model_mean.eval()}")

            log_model_mean = at.log(model_mean)

            print(f"log model mean after: {log_model_mean.eval()}")



            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Lognormal('Y_obs', mu=log_model_mean, sigma=sigma, observed=F)
            #Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            ##print(f"Y_obs logged: {Y_obs.eval()}")
            print(f"Y_obs: {Y_obs.eval()}")

            # Print initial parameter values and logp values for debugging
            initial_values = bayes_model.initial_point()
            ##initial_logp = bayes_model.logp(initial_values)

            print(f"Initial parameter values: {initial_values}")
            ##print(f"Initial logp values: {initial_logp}")

            # Posterior distribution
            idata = pm.sample(100, tune=100, chains=2, cores=1)


        #print(f"idata type inside function: {type(idata)}")

        mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        print(f"mu_hat_np shape: {mu_hat_np.shape}")
        print(f"M_hat_np shape: {M_hat_np.shape}")

        var_names = diagonal_var_names + off_diagonal_var_names

        # Plot and save posterior results
        self.plot_posterior_a(idata, mu_hat_np, M_hat_np, diagonal_var_names, off_diagonal_var_names)

        print(f"idata type after posterior extraction: {type(idata)}")

        return idata, var_names













    def run_bayes_gLV_shrinkage(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model with Horseshoe prior for shrinkage

        Returns:
            idata: The posterior inference data


        """

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M
        DA = self.DA
        DA0 = self.DA0
        N = self.N
        noise_stddev = self.noise_stddev

        # Print the values to debug
        print(f"DA: {DA}, DA0: {DA0}, N: {N}, noise_stddev: {noise_stddev}")

        num_species = F.shape[1]

        bayes_model = pm.Model()
        with bayes_model:
            sigma = pm.HalfNormal('sigma', sigma=1, shape=(1,))  # Same sigma for all responses

            # If mu_value is provided, use it as a fixed value; otherwise, define it as a prior
            if mu_value is not None:
                mu = mu_value
            else:
                mu = pm.Uniform('mu', lower=0.0, upper=0.05)

            # mu_hat = pm.HalfNormal('mu_hat', sigma=1, shape=(1, 5))
            mu_hat = pm.TruncatedNormal('mu_hat',mu=mu,sigma=0.5,lower=0,shape=(1,num_species))

            # M_ii is constrained to be negative
            # M_ii_hat_p = pm.HalfNormal('M_ii_hat_p', sigma=0.1, shape=(5,))
            # M_ii_hat = pm.Deterministic('M_ii_hat', -M_ii_hat_p)
            M_ii_hat = pm.TruncatedNormal('M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(num_species,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy("lam", beta=1, shape=(num_species, num_species-1))
            M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=tau * lam *
                                 at.sqrt(c2 / (c2 + tau ** 2 * lam ** 2)), shape=(num_species, num_species-1))

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((num_species, num_species))
            M_hat_vals = at.set_subtensor(
                M_hat_vals[at.arange(num_species), at.arange(num_species)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(num_species)[:, None], np.delete(
                np.arange(num_species), -1)], M_ij_hat)  # set off-diagonal

            # Save the combined matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(
                X, pm.math.concatenate([M_hat_vals, mu_hat], axis=0))


            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # Posterior distribution
            idata = pm.sample(100, tune=100, chains=2, cores=1)
            #idata = pm.sample(100, tune=100, chains=2, cores=1, init='adapt_diag')

        mu_hat_np = idata.posterior['mu_hat'].mean(dim=('chain', 'draw')).values.flatten()
        M_hat_np = idata.posterior['M_hat'].mean(dim=('chain', 'draw')).values

        print(f"mu_hat_np shape: {mu_hat_np.shape}")
        print(f"M_hat_np shape: {M_hat_np.shape}")

        # Plot and save posterior results
        self.plot_posterior_a(idata, mu_hat_np, M_hat_np)

        return idata

        # Plot and save posterior results
        #self.plot_posterior(idata, mu, M)

        #return idata

        # print summary
        #summary = az.summary(idata, var_names=["mu_hat","M_ii_hat","M_ij_hat","M_hat","sigma"])
        #print(summary[["mean", "sd", "r_hat"]])

        # Write posterior samples to file
        #az.to_netcdf(idata, 'model_posterior.nc')

    def run_bayes_gLV_shrinkage_pert(self) -> None:
        """
        This function infers the parameters for the Bayesian gLV model with Horseshoe prior for shrinkage,
        and including perturbations

        Returns:
            idata: The posterior inference data


        """

        if self.X is None or self.F is None or self.mu is None or self.M is None:
            raise ValueError("Parameters must all be provided.")

        # data = self.data
        X = self.X
        F = self.F
        mu = self.mu
        M = self.M
        DA = self.DA
        DA0 = self.DA0
        N = self.N
        noise_stddev = self.noise_stddev
        epsilon = self.epsilon



        bayes_model = pm.Model()
        with bayes_model:
            sigma = pm.HalfNormal(
                'sigma', sigma=1, shape=(
                    1,))  # Same sigma for all responses

            mu_hat = pm.TruncatedNormal('mu_hat', mu=1.0, sigma=0.5, lower=0, shape=(1,5))

            epsilon_hat = pm.Normal('epsilon_hat', mu=0, sigma=1.0, shape=(1,5))

            # M_ii is constrained to be negative
            M_ii_hat = pm.TruncatedNormal(
                'M_ii_hat', mu=-0.1, sigma=0.1, upper=0, shape=(5,))

            # M_ij is is unconstrained but placed under horseshoe prior
            tau0 = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
            c2 = pm.InverseGamma("c2", 2, 1)
            tau = pm.HalfCauchy("tau", beta=tau0)
            lam = pm.HalfCauchy("lam", beta=1, shape=(5, 4))
            M_ij_hat = pm.Normal('M_ij_hat', mu=0, sigma=tau * lam *
                                 at.sqrt(c2 / (c2 + tau ** 2 * lam ** 2)), shape=(5, 4))

            # Combine values
            # start with an all-zero matrix of the correct shape
            M_hat_vals = at.zeros((5, 5))
            M_hat_vals = at.set_subtensor(
                M_hat_vals[at.arange(5), at.arange(5)], M_ii_hat)  # set diagonal
            M_hat_vals = at.set_subtensor(M_hat_vals[at.arange(5)[:, None], np.delete(
                np.arange(5), -1)], M_ij_hat)  # set off-diagonal

            # Save the combined interaction matrix as a deterministic variable
            M_hat = pm.Deterministic('M_hat', M_hat_vals)

            # Expected value of outcome (linear model)
            model_mean = pm.math.dot(X, pm.math.concatenate(
                [M_hat_vals, mu_hat, epsilon_hat], axis=0))

            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=model_mean, sigma=sigma, observed=F)

            # Posterior distribution
            idata = pm.sample(1000, tune=2000, chains=4, cores=4)

            # Plot and save posterior results
        self.plot_posterior_pert(idata, mu, M, epsilon)

        return idata



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


    def plot_posterior_a(self, idata, mu_hat_np, M_hat_np, diagonal_var_names, off_diagonal_var_names):
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
            var_names=diagonal_var_names,
            ref_val=np.diag(M_hat_np).tolist()
        )
        plt.savefig("plot-posterior-Mii.pdf")
        plt.show()
        plt.close()

        mask = ~np.eye(M_hat_np.shape[0], dtype=bool)
        M_ij = M_hat_np[mask]
        az.plot_posterior(
            idata,
            var_names=off_diagonal_var_names,
            ref_val=M_ij.flatten().tolist()
        )
        plt.savefig("plot-posterior-Mij.pdf")
        plt.show()
        plt.close()



    def plot_posterior_pert(self, idata, mu, M, epsilon):
        """
                Plots the posterior distributions and saves the plots to files.

                Args:
                     idata: The posterior inference data.
                     mu (np.ndarray): The growth rates matrix.
                     M (np.ndarray): The interaction matrix.
                     epsilon (np.ndarray): The perturbation matrix
                       """

        az.plot_posterior(
            idata,
            var_names=["mu_hat"],
            ref_val=mu.flatten().tolist())
        plt.savefig("plot-posterior-mu.pdf")

        az.plot_posterior(
            idata,
            var_names=["epsilon_hat"],
            ref_val=epsilon.flatten().tolist())
        plt.savefig("plot-posterior-eps.pdf")

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

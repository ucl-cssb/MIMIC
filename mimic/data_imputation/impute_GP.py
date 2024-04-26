# mimic/data_imputation/gp_impute.py

import re
from bs4 import ResultSet
from mimic.data_imputation.base_imputator import BaseImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple

import gpflow as gpf
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import reduce_in_tests
from gpflow.kernels import Kernel
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")


class GPImputer(BaseImputer):
    """
    Gaussian Process Imputer using GPFlow for imputing missing data points in datasets.
    Automatically selects an optimal kernel based on the provided dataset.
    """

    def __init__(self):
        """
        Initializes the GPImputer without a pre-defined kernel.
        The optimal kernel is selected based on the dataset.
        """
        super().__init__()
        self.model = None

    def _select_kernel(self, X_train: np.ndarray, Y_train: np.ndarray) -> gpf.kernels.Kernel:
        """
        Automatically selects an optimal kernel for the Gaussian Process based on the provided data.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        :return: An optimal GPFlow kernel instance.
        """

        kernels = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
                   gpf.kernels.Cosine, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
        log_marginal_likelihoods = []

        results = []
        for kernel in kernels:
            model = gpf.models.GPR(data=(X_train, Y_train), kernel=kernel())
            gpf.optimizers.Scipy().minimize(model.training_loss,
                                            variables=model.trainable_variables)
            log_marginal_likelihoods.append(
                model.log_marginal_likelihood().numpy())

        return kernels[np.argmax(log_marginal_likelihoods)]()

    def count_params(self, m):
        p_dict = parameter_dict(m.trainable_parameters)
        # p_dict = parameter_dict(m)
        p_count = 0
        for val in p_dict.values():
            if len(val.shape) == 0:
                p_count = p_count + 1
            else:
                p_count = p_count + math.prod(val.shape)

        return p_count

    # This is for model selection: the higher the BIC the better the model
    # This is the Bayesian Information Criterion (BIC) (Schwarz, 1978) which is a penalized version of the log marginal likelihood (F)
    # This is also shown in lloyd_2014_AutomaticConstructionNaturalLanguage.
    def get_BIC(self, m, F, n):
        # Calculate the Bayesian Information Criterion (BIC)
        # `F` represents the likelihood of the model.
        # `k` is the number of parameters in the model.
        # `n` is the number of data points (sample size).
        # `-2 * np.log(F)` penalizes the negative log likelihood, enhancing the effect of model fit.
        # `k * np.log(n)` adds a penalty that grows with more parameters and larger data size,
        # discouraging overfitting by complex models.
        k = self.count_params(m)
        # QUESTION: Should we use -2 * F + k * np.log(n) or -2 * np.log(F) + k * np.log(n)?
        # return -2 * F + k * np.log(n)
        return -2 * np.log(F) + k * np.log(n)

    # Define the function that will augment the data for the multi-output Gaussian Process model

    def augmentData(self, x, y, p):
        """
        Augments the input data x and y by adding p additional columns to them.

        Parameters:
        x (numpy.ndarray): The input data to be augmented.
        y (numpy.ndarray): The output data to be augmented.
        p (int): The number of additional columns to add.

        Returns:
        numpy.ndarray, numpy.ndarray: The augmented input and output data.
        """
        # Initialize the augmented data
        X_aug = np.hstack((x, np.zeros((x.shape[0], 1))))
        Y_aug = np.hstack((y[:, 0].reshape(-1, 1), np.zeros((y.shape[0], 1))))

        # Add p additional columns
        for i in range(1, p):
            X_aug = np.vstack(
                (X_aug, np.hstack((x, i*np.ones((x.shape[0], 1))))))
            Y_aug = np.vstack(
                (Y_aug, np.hstack((y[:, i].reshape(-1, 1), i*np.ones((y.shape[0], 1))))))

        return X_aug, Y_aug

    # This function is used to optimize the model with scipy
    # QUESTION: Should we use lmfit instead of scipy?
    def optimize_model_with_scipy(self, model, X, Y):

        optimizer = gpf.optimizers.Scipy()
        MAXITER = 5000  # FIXME change this to be user-defined
        if X is not None and Y is not None:
            return optimizer.minimize(
                # QUESTION: Should this be compile=True?
                model.training_loss_closure((X, Y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                # options={"disp": 50, "maxiter": MAXITER},
                options={"maxiter": MAXITER},
            )
        else:
            return optimizer.minimize(
                model.training_loss,
                variables=model.trainable_variables,
                method="l-bfgs-b",
                # options={"disp": 50, "maxiter": MAXITER},
                options={"maxiter": MAXITER},
            )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, kernel, p: int) -> None:
        """
        Fits the GPR model using the optimal kernel to the training data.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        """

        if p > 1:
            # Here do coregionalization to estimate f(x) = W g(x)
            # https://gpflow.github.io/GPflow/2.8.0/notebooks/advanced/multioutput.html
            # https://gpflow.github.io/GPflow/develop/notebooks/getting_started/mean_functions.html
            # https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7
            # https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/advanced/coregionalisation.html
            # https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/coregionalisation.html
            # This uses SVGP

            # Coregionalization kernel
            L = 1  # rank of the coregionalization matrix FIXME: change this to be user-defined
            # coreg = gpf.kernels.Coregion(input_dim=1, output_dim=p, rank=p)
            coreg = gpf.kernels.Coregion(output_dim=p, rank=L, active_dims=[1])

            kernel = kernel(active_dims=[0]) * coreg

            # multi-output Gaussian Process model
            # m = gpf.models.SVGP(data=(X_train, Y_train),
            # m = gpf.models.SVGP(kernel=kernel, likelihood=gpf.likelihoods.Gaussian(
            # ), inducing_variable=X_train_aug, num_latent_gps=p)
            m = gpf.models.VGP((X_train, Y_train),
                               kernel=kernel, likelihood=gpf.likelihoods.Gaussian())

        else:
            # single-output Gaussian Process model
            m = gpf.models.GPR(data=(X_train, Y_train),
                               kernel=kernel(active_dims=[0]))

        # NOTE: Since this code is using VGP, we use the same optimizer as in the original code
        # However, when we implement the SVGP model, we should use the optimizer and data from the original code
        # reference: https://gpflow.github.io/GPflow/2.9.1/notebooks/advanced/multioutput.html
        res = self.optimize_model_with_scipy(m, None, None)

        bic = self.get_BIC(m, res.fun, X_train.shape[0])

        return m, bic

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts values for new data using the fitted GPR model.

        :param X_new: New data features.
        :return: A tuple of predicted means and variances.
        """
        mean, var = self.model.predict_y(X_new)
        return mean.numpy(), var.numpy()

    def generate_kernel_library(self):
        kernels = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
                   gpf.kernels.Cosine, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
        return kernels

    def impute_missing_values(self, dataset: pd.DataFrame, feature_columns: list, output_columns: list, target_column: str, kernel: str = None) -> pd.DataFrame:
        """
        Imputes missing values in the dataset for the specified target column using the GPR model.

        :param dataset: The dataset with missing values.
        :param feature_columns: List of feature column names.
        :param target_column: The target column name for imputation.
        :return: Dataset with imputed values in the target column.
        """
        # make a copy of the dataset
        dataset = dataset.copy()

        # Check if the target column has missing values
        missing_mask = dataset[target_column].isnull()
        if missing_mask.sum() == 0:
            print("No missing values found in the target column.")
            return dataset

        # FIXME: change this for when we have multiple output columns
        train_data = dataset[~missing_mask]
        missing_data = dataset[missing_mask]

        X_train = train_data[feature_columns].values
        # Y_train = train_data[[target_column]].values
        Y_train = train_data[output_columns].values

        # p is the dimension of the output(s)
        p = Y_train.shape[1]
        if p > 1:
            X_train, Y_train = self.augmentData(X_train, Y_train, p)

        if kernel is None:
            kernels = self.generate_kernel_library()
        elif isinstance(kernel, Kernel):
            kernels = [kernel]
        elif isinstance(kernel, str):
            kernel_map = {
                'RBF': "SquaredExponential",
                'M32': "Matern32",
                'RQ': "RationalQuadratic",
                'Exp': "Exponential",
                'Lin': "Linear",
                'Cos': "Cosine",
                'Poly': "Polynomial",
                'M12': "Matern12",
                'M52': "Matern52",
                'White': "White"
            }
            kernel_library = {
                k.__name__: k for k in self.generate_kernel_library()}
            if kernel in kernel_map:
                kernel_name = kernel_map[kernel]
                if kernel_name in kernel_library:
                    kernels = [kernel_library[kernel_name]]
                else:
                    raise ValueError(
                        f"Unknown kernel '{kernel}'. Available options: {list(self.kernel_map.keys())}")
        else:
            raise ValueError("Invalid kernel provided.")

        results = []
        for kernel in kernels:
            m, bic = self.fit(X_train, Y_train, kernel, p)
            # QUESTION: add mean here? Or always assume the mean is 0?
            results.append([m, bic, kernel])

        # Find the best fitted model
        # by finding the model with the highest BIC
        bestModel = max(results, key=lambda x: x[1])
        self.model = bestModel[0]
        bic = bestModel[1]
        k_L = bestModel[2]

        print(f"Model: {self.model.__class__.__name__}")
        print(f"Best kernel: {k_L}")
        print(f"BIC: {bic}")

        if not missing_data.empty:
            X_missing = missing_data[feature_columns].values
            if p > 1:
                X_missing, _ = self.augmentData(
                    X_missing, Y_train, 1)  # FIXME change 1 to p or target_column
                predicted_means_missing, _ = self.predict(X_missing)
                predicted_means_missing = predicted_means_missing[:, 0]
            else:
                predicted_means_missing, _ = self.predict(X_missing)
            dataset.loc[missing_mask,
                        target_column] = predicted_means_missing.flatten()

        # make full predicted means and variances available for plotting

        extended_dataset = self.predict_extended_range(dataset)

        if p > 1:
            X_new = extended_dataset
            X_new_aug, _ = self.augmentData(X_new, Y_train, 1)
            predicted_means_new, predicted_variances_new = self.predict(
                X_new_aug)
            predicted_means_new = predicted_means_new[:, 0]
            predicted_variances_new = predicted_variances_new[:, 0]
            self.plot_imputed_data(train_data[feature_columns].values, train_data[[target_column]].values, X_missing[:, 0], predicted_means_missing, extended_dataset,
                                   predicted_means_new, predicted_variances_new)
        else:
            predicted_means_new, predicted_variances_new = self.predict(
                extended_dataset)
            self.plot_imputed_data(X_train, Y_train, X_missing, predicted_means_missing, extended_dataset,
                                   predicted_means_new, predicted_variances_new)

        self.data = dataset
        return dataset

    # Extend the data points to be predicted further than the original data points
    # to make the plot look better

    def predict_extended_range(self, dataset, extend_percent=0.1):
        """
        Extends the data points to be predicted further than the original data points
        to make the plot look better. Extends the range by a given percentage.

        :param dataset: The dataset containing the original X values.
        :param extend_percent: The percentage of the range to extend on both ends.
        :return: An array of X values extended beyond the original range.
        """
        # Get the original X values
        original_data = dataset.iloc[:, 0].values

        # Calculate the range of the data
        data_range = original_data.max() - original_data.min()

        # Calculate the extensions
        lower_extension = original_data.min() - (data_range * extend_percent)
        upper_extension = original_data.max() + (data_range * extend_percent)

        # Number of points to add on each side
        num_points = len(original_data)
        extended_lower = np.linspace(
            lower_extension, original_data.min(), num=num_points // 10, endpoint=False)
        extended_upper = np.linspace(
            original_data.max(), upper_extension, num=num_points // 10, endpoint=False)

        # Combine original and extended points
        extended_range = np.concatenate(
            (extended_lower, original_data, extended_upper))

        return extended_range.reshape(-1, 1)

    # plot the original and imputed data

    def plot_imputed_data(self, X_train: np.ndarray, Y_train: np.ndarray, X_missing, predicted_means_missing, X_new: np.ndarray, predicted_means: np.ndarray, predicted_variances: np.ndarray) -> None:
        """
        Plots the original and imputed data points.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        :param X_new: New data features.
        :param Y_new: New data targets.
        :param predicted_means: Predicted means for the new data.
        :param predicted_variances: Predicted variances for the new data.
        """

        plt.figure(figsize=(12, 6))
        # plt.plot(X_missing, np.zeros_like(
        #     X_missing), 'bo', label='Missing Data Points')
        plt.plot(X_missing, predicted_means_missing,
                 'bx', label='Imputed Data')
        # plt.plot(X_missing, Y_new[X_missing.astype(
        #    int)], 'bx', label='Imputed Data')

        plt.plot(X_new, predicted_means,
                 'g-', label='Predicted Function')
        plt.fill_between((X_new).flatten(), predicted_means.flatten() -
                         1.96 * np.sqrt(predicted_variances.flatten()), predicted_means.flatten() + 1.96 * np.sqrt(predicted_variances.flatten()), color='g', alpha=0.1, label='95% Confidence Interval')

        plt.plot(X_train, Y_train, 'ro', label='Training Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

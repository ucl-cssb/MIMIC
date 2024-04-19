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
    def get_BIC(self, m, F, n):
        k = self.count_params(m)
        return -2 * F + k * np.log(n)

    # This function is used to optimize the model with scipy
    # QUESTION: Should we use lmfit instead of scipy?
    def optimize_model_with_scipy(self, model, X, Y):
        optimizer = gpf.optimizers.Scipy()
        MAXITER = 5000  # FIXME change this to be user-defined
        return optimizer.minimize(
            model.training_loss_closure(),
            variables=model.trainable_variables,
            method="l-bfgs-b",
            # options={"disp": 50, "maxiter": MAXITER},
            options={"maxiter": MAXITER},
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, kernel) -> None:
        """
        Fits the GPR model using the optimal kernel to the training data.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        """
        # optimal_kernel = self._select_kernel(X_train, Y_train)
        m = gpf.models.GPR(
            data=(X_train, Y_train), kernel=kernel(active_dims=[0]))
        res = self.optimize_model_with_scipy(m, X_train, Y_train)

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

    def impute_missing_values(self, dataset: pd.DataFrame, feature_columns: list, target_column: str, kernel: str = None) -> pd.DataFrame:
        """
        Imputes missing values in the dataset for the specified target column using the GPR model.

        :param dataset: The dataset with missing values.
        :param feature_columns: List of feature column names.
        :param target_column: The target column name for imputation.
        :return: Dataset with imputed values in the target column.
        """
        # make a copy of the dataset
        dataset = dataset.copy()
        missing_mask = dataset[target_column].isnull()
        if missing_mask.sum() == 0:
            print("No missing values found in the target column.")
            return dataset
        train_data = dataset[~missing_mask]
        missing_data = dataset[missing_mask]

        X_train = train_data[feature_columns].values
        Y_train = train_data[[target_column]].values

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
            m, bic = self.fit(X_train, Y_train, kernel)
            # QUESTION: add mean here? Or always assume the mean is 0?
            results.append([m, bic, kernel])

        # Find the best fitted model
        # by finding the model with the highest BIC
        bestModel = max(results, key=lambda x: x[1])
        self.model = bestModel[0]
        bic = bestModel[1]
        k_L = bestModel[2]

        print(f"Best kernel: {k_L}")
        print(f"BIC: {bic}")

        if not missing_data.empty:
            X_missing = missing_data[feature_columns].values
            predicted_means, _ = self.predict(X_missing)
            dataset.loc[missing_mask,
                        target_column] = predicted_means.flatten()

        # make full predicted means and variances available for plotting

        extended_dataset = self.predict_extended_range(dataset)

        predicted_means_new, predicted_variances_new = self.predict(
            extended_dataset)

        self.plot_imputed_data(X_train, Y_train, X_missing, extended_dataset, dataset.iloc[:, 1].values,
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

    def plot_imputed_data(self, X_train: np.ndarray, Y_train: np.ndarray, X_missing, X_new: np.ndarray, Y_new: np.ndarray, predicted_means: np.ndarray, predicted_variances: np.ndarray) -> None:
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

        plt.plot(X_missing, Y_new[X_missing.astype(
            int)], 'bx', label='Imputed Data')

        plt.plot(X_new, predicted_means,
                 'g-', label='Predicted Function')
        plt.fill_between((X_new).flatten(), predicted_means.flatten() -
                         1.96 * np.sqrt(predicted_variances.flatten()), predicted_means.flatten() + 1.96 * np.sqrt(predicted_variances.flatten()), color='g', alpha=0.1, label='95% Confidence Interval')

        plt.plot(X_train, Y_train, 'ro', label='Training Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

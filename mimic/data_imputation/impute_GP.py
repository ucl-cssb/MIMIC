# mimic/data_imputation/gp_impute.py

import gpflow
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


class GPImputer:
    """
    Gaussian Process Imputer using GPFlow for imputing missing data points in datasets.
    Automatically selects an optimal kernel based on the provided dataset.
    """

    def __init__(self):
        """
        Initializes the GPImputer without a pre-defined kernel.
        The optimal kernel is selected based on the dataset.
        """
        self.model = None

    def _select_kernel(self, X_train: np.ndarray, Y_train: np.ndarray) -> gpflow.kernels.Kernel:
        """
        Automatically selects an optimal kernel for the Gaussian Process based on the provided data.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        :return: An optimal GPFlow kernel instance.
        """

        kernels = [gpflow.kernels.SquaredExponential, gpflow.kernels.Matern32, gpflow.kernels.RationalQuadratic, gpflow.kernels.Exponential, gpflow.kernels.Linear,
                   gpflow.kernels.Cosine, gpflow.kernels.Polynomial, gpflow.kernels.Matern12, gpflow.kernels.Matern52, gpflow.kernels.White]
        log_marginal_likelihoods = []

        for kernel in kernels:
            model = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel())
            gpflow.optimizers.Scipy().minimize(model.training_loss,
                                               variables=model.trainable_variables)
            log_marginal_likelihoods.append(
                model.log_marginal_likelihood().numpy())

        return kernels[np.argmax(log_marginal_likelihoods)]()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        """
        Fits the GPR model using the optimal kernel to the training data.

        :param X_train: Training data features.
        :param Y_train: Training data targets.
        """
        optimal_kernel = self._select_kernel(X_train, Y_train)
        self.model = gpflow.models.GPR(
            data=(X_train, Y_train), kernel=optimal_kernel)
        gpflow.optimizers.Scipy().minimize(self.model.training_loss,
                                           variables=self.model.trainable_variables)

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts values for new data using the fitted GPR model.

        :param X_new: New data features.
        :return: A tuple of predicted means and variances.
        """
        mean, var = self.model.predict_y(X_new)
        return mean.numpy(), var.numpy()

    def impute_missing_values(self, dataset: pd.DataFrame, feature_columns: list, target_column: str) -> pd.DataFrame:
        """
        Imputes missing values in the dataset for the specified target column using the GPR model.

        :param dataset: The dataset with missing values.
        :param feature_columns: List of feature column names.
        :param target_column: The target column name for imputation.
        :return: Dataset with imputed values in the target column.
        """
        missing_mask = dataset[target_column].isnull()
        train_data = dataset[~missing_mask]
        missing_data = dataset[missing_mask]

        X_train = train_data[feature_columns].values
        Y_train = train_data[[target_column]].values
        self.fit(X_train, Y_train)

        if not missing_data.empty:
            X_missing = missing_data[feature_columns].values
            predicted_means, _ = self.predict(X_missing)
            dataset.loc[missing_mask,
                        target_column] = predicted_means.flatten()

        # make full predicted means and variances available for plotting

        predicted_means_new, predicted_variances_new = self.predict(
            self.predict_extended_range(dataset))

        self.plot_imputed_data(X_train, Y_train, X_missing, dataset.iloc[:, 0].values, dataset.iloc[:, 1].values,
                               predicted_means_new, predicted_variances_new)

        return dataset

    # Extend the data points to be predicted further than the original data points
    # to make the plot look better

    def predict_extended_range(self, dataset):
        # Get the original data
        original_data = dataset.iloc[:, 0].values.reshape(-1, 1)

        # Calculate the range of the data
        data_range = original_data.max() - original_data.min()

        # Calculate the extension
        extension = data_range * 0.1

        return np.linspace(
            original_data.min() - extension,
            original_data.max() + extension,
            len(original_data),
        ).reshape(-1, 1)

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
        plt.plot(X_missing, np.zeros_like(
            X_missing), 'bo', label='Missing Data Points')

        plt.plot(X_missing, Y_new[X_missing.astype(
            int)], 'bx', label='Imputed Data')

        # extension = (X_new.max() - X_new.min()) * 0.1

        plt.plot(X_new, predicted_means,
                 'g-', label='Predicted Function')
        plt.fill_between((X_new).flatten(), predicted_means.flatten() -
                         1.96 * np.sqrt(predicted_variances.flatten()), predicted_means.flatten() + 1.96 * np.sqrt(predicted_variances.flatten()), color='g', alpha=0.1, label='95% Confidence Interval')

        plt.plot(X_train, Y_train, 'ro', label='Training Data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

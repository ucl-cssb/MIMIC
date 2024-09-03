# tests/test_gp_imputer.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from mimic.data_imputation.impute_GP import GPImputer


def test_gp_imputer_initialization():
    imputer = GPImputer()
    assert imputer.model is None


@patch('mimic.data_imputation.gp_impute.gpf.models.GPR')
@patch('mimic.data_imputation.gp_impute.GPImputer.optimize_model_with_scipy')
@patch('mimic.data_imputation.gp_impute.GPImputer.get_BIC')
def test_fit_single_output(mock_get_BIC, mock_optimize_model, mock_GPR):
    imputer = GPImputer()
    X_train = np.array([[1], [2], [3]])
    Y_train = np.array([[1], [2], [3]])
    kernel = MagicMock()

    mock_GPR.return_value = MagicMock()
    mock_optimize_model.return_value = MagicMock(fun=0.5)
    mock_get_BIC.return_value = -100.0

    model, bic = imputer.fit(X_train, Y_train, kernel, p=1)

    mock_GPR.assert_called_once()
    mock_optimize_model.assert_called_once()
    mock_get_BIC.assert_called_once()

    assert model is not None
    assert bic == -100.0


def test_augment_data():
    imputer = GPImputer()
    X = np.array([[1], [2], [3]])
    Y = np.array([[10, 20], [30, 40], [50, 60]])
    p = 2

    X_aug, Y_aug = imputer.augmentData(X, Y, p)

    assert X_aug.shape == (6, 2)
    assert Y_aug.shape == (6, 2)


@patch('mimic.data_imputation.gp_impute.gpf.kernels.SquaredExponential')
@patch('mimic.data_imputation.gp_impute.GPImputer.fit')
@patch('mimic.data_imputation.gp_impute.GPImputer.predict')
@patch('mimic.data_imputation.gp_impute.GPImputer.plot_imputed_data')
def test_impute_missing_values(mock_plot, mock_predict, mock_fit, mock_kernel):
    imputer = GPImputer()
    dataset = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Target': [1.0, np.nan, 3.0, np.nan, 5.0]
    })
    feature_columns = ['Feature1']
    output_columns = ['Target']
    target_column = 'Target'

    mock_fit.return_value = (MagicMock(), -100.0)
    mock_predict.side_effect = [(np.array([[2.0], [4.0]]), np.array([[0.1], [0.2]])),
                                (np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), np.array([[0.1], [0.1], [0.1], [0.1], [0.1]]))]

    imputed_dataset = imputer.impute_missing_values(
        dataset, feature_columns, output_columns, target_column, kernel="RBF")

    assert 'Target' in imputed_dataset.columns
    assert imputed_dataset['Target'].notnull().all()

    # Ensure plot was mocked and not actually called
    mock_plot.assert_called_once()

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.model_selection import RepeatedKFold, cross_val_score
from mimic.model_infer.infer_gMLV_lm import *


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 3)
    return X, y


def test_ridge1_initialization():
    model = Ridge1(alphas=[0.01, 0.1], num_species=5)
    assert model.alphas == [0.01, 0.1]
    assert model.num_species == 5
    assert model.coef_ is None


def test_ridge1_fit(mock_data):
    X, y = mock_data
    model = Ridge1(alphas=[0.01, 0.1], num_species=3)
    model.fit(X, y)
    assert model.coef_ is not None
    # Adjust based on X's shape
    assert model.coef_.shape == (model.num_species, X.shape[1])


def test_ridge1_predict(mock_data):
    X, y = mock_data
    model = Ridge1(alphas=[0.01, 0.1], num_species=3)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (10, 3)


def test_ridge1_not_fitted_error(mock_data):
    X, _ = mock_data
    model = Ridge1()
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.predict(X)


def test_ridge2_initialization():
    model = Ridge2(alphas=[0.01, 0.1, 0.01], num_species=5, num_pert=2)
    assert model.alphas == [0.01, 0.1, 0.01]
    assert model.num_species == 5
    assert model.num_pert == 2
    assert model.coef_ is None


def test_ridge2_fit(mock_data):
    X, y = mock_data
    model = Ridge2(alphas=[0.01, 0.1, 0.01], num_species=3, num_pert=1)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.coef_.shape == (3, 5)  # Assuming the output is (3, 5)


def test_ridge2_predict(mock_data):
    X, y = mock_data
    model = Ridge2(alphas=[0.01, 0.1, 0.01], num_species=3, num_pert=1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (10, 3)


def test_ridge2_not_fitted_error(mock_data):
    X, _ = mock_data
    model = Ridge2()
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.predict(X)


def test_ridge_fit(mock_data):
    X, y = mock_data
    alphas = [0.1, 0.1]
    num_species = 3
    coefficients = ridge_fit(X, y, alphas, num_species)
    assert coefficients.shape == (3, 4)


def test_ridge_fit_pert(mock_data):
    X, y = mock_data
    alphas = [0.1, 0.1, 0.1]
    num_species = 3
    num_pert = 1
    coefficients = ridge_fit_pert(X, y, alphas, num_species, num_pert)
    assert coefficients.shape == (3, 5)


def test_linearize_time_course_16S():
    yobs = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    times = np.array([0.0, 1.0])
    tX, tF = linearize_time_course_16S(yobs, times)
    assert tX.shape == (1, 4)  # 1 time step, 3 species + 1 constant
    assert tF.shape == (1, 3)  # 1 time step, 3 species


def test_do_final_fit_Ridge1(mock_data):
    X, y = mock_data
    num_species = 3
    a0 = 0.1
    a1 = 0.1
    mu_h, M_h = do_final_fit_Ridge1(X, y, num_species, a0, a1)
    assert len(mu_h) == num_species
    assert len(M_h) == num_species
    assert len(M_h[0]) == num_species


@pytest.mark.parametrize("X_shape,y_shape,num_species", [
    ((10, 3), (10, 3), 3),
    ((15, 4), (15, 4), 4),
])
def test_ridge1_fit_with_different_shapes(X_shape, y_shape, num_species):
    X = np.random.rand(*X_shape)
    y = np.random.rand(*y_shape)
    model = Ridge1(alphas=[0.01, 0.1], num_species=num_species)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.coef_.shape == (num_species, X.shape[1] + 1)


@pytest.mark.parametrize("X_shape,y_shape,num_species,num_pert", [
    ((10, 4), (10, 4), 3, 1),
    ((15, 5), (15, 5), 4, 2),
])
def test_ridge2_fit_with_different_shapes(X_shape, y_shape, num_species, num_pert):
    X = np.random.rand(*X_shape)
    y = np.random.rand(*y_shape)
    model = Ridge2(alphas=[0.01, 0.1, 0.01],
                   num_species=num_species, num_pert=num_pert)
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.coef_.shape == (num_species, X.shape[1] + 1 + num_pert)


@patch('sklearn.model_selection.cross_val_score')
def test_fit_alpha_Ridge1(mock_cross_val_score, mock_data):
    X, y = mock_data
    mock_cross_val_score.return_value = np.random.rand(10)
    num_species = 3
    a0, a1 = fit_alpha_Ridge1(X, y, num_species, 3, 3)
    assert a0 > 0
    assert a1 > 0


@patch('sklearn.model_selection.cross_val_score')
def test_fit_alpha_Ridge2(mock_cross_val_score, mock_data):
    X, y = mock_data
    mock_cross_val_score.return_value = np.random.rand(10)
    num_species = 3
    num_pert = 1
    a0, a1, a2 = fit_alpha_Ridge2(X, y, num_species, num_pert, 3, 3, 3)
    assert a0 > 0
    assert a1 > 0
    assert a2 > 0

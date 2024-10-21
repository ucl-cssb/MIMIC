import numpy as np
import pytest
from mimic.model_infer.infer_gMLV_lm import Ridge1, Ridge2, ridge_fit, ridge_fit_pert

# ----------------------------
# Mock Data for Ridge1 (Ridge1)
# ----------------------------
# Parameters: num_species=2
# Expected X shape: (n_samples, num_species + 1) => (3, 3)
X_mock_ridge1 = np.array([
    [1, 2, 3],  # Sample 1
    [4, 5, 6],  # Sample 2
    [7, 8, 9]   # Sample 3
])

y_mock = np.array([
    [1, 2],  # Target 1 for Sample 1 and 2
    [3, 4],
    [5, 6]
])

# ---------------------------------
# Mock Data for Ridge2 (Ridge2)
# ---------------------------------
# Parameters: num_species=2, num_pert=1
# Expected X shape: (n_samples, num_species + num_pert + 1) => (3, 4)
X_mock_ridge2 = np.array([
    [1, 2, 3, 4],  # Sample 1
    [5, 6, 7, 8],  # Sample 2
    [9, 10, 11, 12]  # Sample 3
])

# ----------------------------
# Test Cases for Ridge1
# ----------------------------


def test_ridge1_initialization():
    """Test initialization of Ridge1."""
    model = Ridge1(alphas=[0.5, 0.5], num_species=2)
    assert model.alphas == [0.5, 0.5], "Alphas not set correctly."
    assert model.num_species == 2, "Number of species not set correctly."


def test_ridge1_fit():
    """Test fitting Ridge1 with mock data."""
    model = Ridge1(alphas=[0.5, 0.5], num_species=2)
    model.fit(X_mock_ridge1, y_mock)
    assert model.coef_ is not None, "Coefficients not set after fitting."
    assert model.coef_.shape == (
        2, 3), f"Expected coef shape (2,3), got {model.coef_.shape}."


def test_ridge1_predict():
    """Test prediction using Ridge1."""
    model = Ridge1(alphas=[0.5, 0.5], num_species=2)
    model.fit(X_mock_ridge1, y_mock)
    y_pred = model.predict(X_mock_ridge1)
    assert y_pred.shape == y_mock.shape, f"Expected prediction shape {y_mock.shape}, got {y_pred.shape}."


def test_ridge1_unfitted_predict():
    """Test that Ridge1 raises an error when predicting before fitting."""
    model = Ridge1(alphas=[0.5, 0.5], num_species=2)
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.predict(X_mock_ridge1)

# ----------------------------
# Test Cases for Ridge2
# ----------------------------


def test_ridge2_initialization():
    """Test initialization of Ridge2."""
    model = Ridge2(alphas=[0.1, 0.2, 0.3], num_species=2, num_pert=1)
    assert model.alphas == [0.1, 0.2, 0.3], "Alphas not set correctly."
    assert model.num_species == 2, "Number of species not set correctly."
    assert model.num_pert == 1, "Number of perturbations not set correctly."


def test_ridge2_fit():
    """Test fitting Ridge2 with mock data."""
    model = Ridge2(alphas=[0.1, 0.2, 0.3], num_species=2, num_pert=1)
    model.fit(X_mock_ridge2, y_mock)
    assert model.coef_ is not None, "Coefficients not set after fitting."
    assert model.coef_.shape == (
        2, 4), f"Expected coef shape (2,4), got {model.coef_.shape}."


def test_ridge2_predict():
    """Test prediction using Ridge2."""
    model = Ridge2(alphas=[0.1, 0.2, 0.3], num_species=2, num_pert=1)
    model.fit(X_mock_ridge2, y_mock)
    y_pred = model.predict(X_mock_ridge2)
    assert y_pred.shape == y_mock.shape, f"Expected prediction shape {y_mock.shape}, got {y_pred.shape}."


def test_ridge2_unfitted_predict():
    """Test that Ridge2 raises an error when predicting before fitting."""
    model = Ridge2(alphas=[0.1, 0.2, 0.3], num_species=2, num_pert=1)
    with pytest.raises(ValueError, match="Model is not fitted yet."):
        model.predict(X_mock_ridge2)

# ----------------------------
# Test Cases for ridge_fit and ridge_fit_pert
# ----------------------------


def test_ridge_fit_function():
    """Test the ridge_fit function."""
    result = ridge_fit(X_mock_ridge1.T, y_mock.T,
                       alphas=[0.1, 0.1], num_species=2)
    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert result.shape == (
        2, 3), f"Expected result shape (2,3), got {result.shape}."


def test_ridge_fit_pert_function():
    """Test the ridge_fit_pert function."""
    result = ridge_fit_pert(X_mock_ridge2.T, y_mock.T, alphas=[
                            0.1, 0.2, 0.3], num_species=2, num_pert=1)
    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert result.shape == (
        2, 4), f"Expected result shape (2,4), got {result.shape}."

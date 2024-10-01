import pytest
import numpy as np
from mimic.model_infer.multi_penalty_lasso import MultiPenaltyLasso, fit_alpha_MPLasso
from unittest.mock import patch


@pytest.fixture
def mock_data():
    """Fixture to provide mock data for testing."""
    np.random.seed(42)
    X = np.random.rand(10, 3)  # 10 samples, 3 features
    y = np.random.rand(10, 2)  # 10 samples, 2 target variables
    return X, y


def test_initialization():
    """Test the initialization of the MultiPenaltyLasso class."""
    alpha = np.array([0.1, 0.5, 1.0])
    model = MultiPenaltyLasso(alpha=alpha)
    assert np.all(model.alpha == alpha)
    assert model.coef_ is None


def test_non_zero_penalties(mock_data):
    """Test non-zero penalties fit function."""
    X, y = mock_data
    alpha = np.array([0.1, 0.5, 1.0])
    model = MultiPenaltyLasso(alpha=alpha)
    model.fit(X, y)
    assert model.coef_ is not None
    # Coefficients should match targets x features (since coef_.shape is reversed in this case)
    assert model.coef_.shape == (y.shape[1], X.shape[1])


def test_penalised_lasso(mock_data):
    """Test penalized lasso fit when some alpha values are zero."""
    X, y = mock_data
    alpha = np.array([0.0, 0.5, 1.0])
    model = MultiPenaltyLasso(alpha=alpha)
    model.fit(X, y)
    assert model.coef_ is not None
    # Ensure correct coefficient shapes (features x targets)
    # Adjusted to (features, targets)
    assert model.coef_.shape == (X.shape[1], y.shape[1])


def test_predict(mock_data):
    """Test prediction after fitting."""
    X, y = mock_data
    alpha = np.array([0.1, 0.5, 1.0])
    model = MultiPenaltyLasso(alpha=alpha)
    model.fit(X, y)
    predictions = model.predict(X)
    # Prediction shape should match the target shape
    assert predictions.shape == y.shape


def test_predict_not_fitted_error(mock_data):
    """Test that predict raises an error if model is not fitted."""
    X, _ = mock_data
    model = MultiPenaltyLasso(alpha=np.array([0.1, 0.5, 1.0]))
    with pytest.raises(ValueError, match="Model not fitted"):
        model.predict(X)


@patch('mimic.model_infer.multi_penalty_lasso.cross_val_score')
def test_fit_alpha_MPLasso(mock_cross_val_score, mock_data):
    """Test the alpha fitting function for MultiPenaltyLasso."""
    X, y = mock_data
    mock_cross_val_score.return_value = np.random.rand(5)  # Mock cross

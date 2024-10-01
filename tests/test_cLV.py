import pytest
import numpy as np
from mimic.model_infer.cLV import (
    CompositionalLotkaVolterra,
    choose_denom,
    construct_alr,
    estimate_elastic_net_regularizers_cv,
    elastic_net_clv,
    ridge_regression_clv,
    predict,
    compute_prediction_error,
    estimate_relative_abundances
)


@pytest.fixture
def example_data():
    P = [np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])]
    T = [np.array([0, 1, 2])]
    U = [np.array([[0], [0], [0]])]
    return P, T, U


@pytest.fixture
def clv_model(example_data):
    P, T, U = example_data
    return CompositionalLotkaVolterra(P=P, T=T, U=U)


def test_initialization(clv_model, example_data):
    P, T, U = example_data
    assert clv_model.P == P
    assert clv_model.T == T
    assert clv_model.U == U
    assert clv_model.X is not None
    assert clv_model.denom is not None
    expected_no_effects = U is None or np.all([np.all(u == 0) for u in U])

    assert clv_model.no_effects == expected_no_effects


def test_predict_function(mocker):
    mock_compute_rel_abun = mocker.patch(
        'mimic.model_infer.cLV.compute_rel_abun', return_value=np.array([[0.1, 0.9, 0.0]]))
    mock_solve_ivp = mocker.patch(
        'scipy.integrate.solve_ivp', return_value=mocker.Mock(y=np.array([[0.2, 0.8, 0.0]])))

    x = np.array([[0.1, 0.9]])
    p = np.array([[0.1, 0.9, 0.0]])
    u = np.array([[0]])
    times = np.array([0, 1])

    A = np.array([[0.1, 0.2, 0.0]])
    g = np.array([0.1])
    B = np.array([[0.1]])

    pred = predict(x, p, u, times, A, g, B, denom=0)
    assert pred.shape == (2, 3)


def test_get_set_regularizers(clv_model):
    clv_model.set_regularizers(alpha=0.1, r_A=0.2, r_g=0.3, r_B=0.4)
    assert clv_model.get_regularizers() == (0.1, 0.2, 0.3, 0.4)


def test_train(mocker, clv_model):
    P = [
        np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
        np.array([[0.2, 0.2, 0.6], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    ]
    T = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    U = [np.array([[0], [0], [0]]), np.array([[0], [0], [0]])]

    clv_model = CompositionalLotkaVolterra(P=P, T=T, U=U)

    mock_estimate_regularizers = mocker.patch(
        'mimic.model_infer.cLV.estimate_elastic_net_regularizers_cv',
        return_value=(0.1, 0.2, 0.3, 0.4)
    )
    mock_elastic_net_clv = mocker.patch(
        'mimic.model_infer.cLV.elastic_net_clv',
        return_value=(np.array([[1, 0, 0], [0, 1, 0]]),
                      np.array([0.1, 0.2]), np.array([[0.3], [0.4]]))
    )

    clv_model.train(verbose=True)
    assert clv_model.A is not None
    assert clv_model.g is not None
    assert clv_model.B is not None


def test_get_params(clv_model):
    clv_model.A = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    clv_model.g = np.array([0.1, 0.2])
    clv_model.B = np.array([[0.1], [0.2]])
    A, g, B = clv_model.get_params()
    assert A.shape == clv_model.A.shape
    assert g.shape == clv_model.g.shape
    assert B.shape == clv_model.B.shape


def test_choose_denom():
    P = [np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])]
    denom = choose_denom(P)
    assert denom == 0 or denom == 1 or denom == 2


def test_construct_alr():
    P = [np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])]
    denom = 0
    alr = construct_alr(P, denom)
    assert len(alr) == 1
    assert alr[0].shape == (3, 2)


def test_estimate_elastic_net_regularizers_cv(mocker):
    P = [
        np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
        np.array([[0.2, 0.2, 0.6], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    ]
    T = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    U = [np.array([[0], [0], [0]]), np.array([[0], [0], [0]])]

    X = construct_alr(P, denom=0)
    mock_elastic_net_clv = mocker.patch(
        'mimic.model_infer.cLV.elastic_net_clv',
        return_value=(np.array([[1, 0, 0], [0, 1, 0]]),
                      np.array([0.1, 0.2]), np.array([[0.3], [0.4]]))
    )
    best_r = estimate_elastic_net_regularizers_cv(X, P, U, T, denom=0, folds=2)
    assert isinstance(best_r, tuple)
    assert len(best_r) == 4


def test_elastic_net_clv():
    P = [np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])]
    T = [np.array([0, 1, 2])]
    U = [np.array([[0], [0], [0]])]
    X = construct_alr(P, denom=0)
    Q_inv = np.eye(X[0].shape[1])
    A, g, B = elastic_net_clv(
        X, P, U, T, Q_inv, alpha=0.1, r_A=0.2, r_g=0.3, r_B=0.4)
    assert A.shape == (2, 3)
    assert g.shape == (2,)
    assert B.shape == (2, 1)


def test_ridge_regression_clv():
    P = [np.array([[0.1, 0.3, 0.6], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])]
    T = [np.array([0, 1, 2])]
    U = [np.array([[0], [0], [0]])]
    X = construct_alr(P, denom=0)
    A, g, B = ridge_regression_clv(X, P, U, T, r_A=0.1, r_g=0.2, r_B=0.3)
    assert A.shape == (2, 3)
    assert g.shape == (2,)
    assert B.shape == (2, 1)


def test_compute_prediction_error(example_data):
    P, T, U = example_data
    X = construct_alr(P, denom=0)
    A = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    g = np.array([0.1, 0.2])
    B = np.array([[0.1], [0.2]])
    err = compute_prediction_error(X, P, U, T, A, g, B, denom_ids=0)
    assert isinstance(err, float)


def test_estimate_relative_abundances():
    Y = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
    P = estimate_relative_abundances(Y)
    assert len(P) == 1
    assert P[0].shape == (3, 3)

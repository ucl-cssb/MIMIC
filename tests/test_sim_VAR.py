import numpy as np
import pytest
from mimic.model_simulate.sim_VAR import sim_VAR


def test_var_initialization():
    model = sim_VAR()
    assert model.model == "VAR"
    assert model.n_obs is None
    assert model.coefficients is None
    assert model.initial_values is None
    assert model.noise_stddev is None
    assert model.output is None
    assert model.dataM is None
    assert model.coefficientsM is None
    assert model.initial_valuesM is None


def test_set_parameters():
    model = sim_VAR()
    model.set_parameters(
        n_obs=100,
        coefficients=[[0.5]],
        initial_values=[[1]],
        noise_stddev=0.1,
        output="show"
    )

    assert model.n_obs == 100
    assert model.coefficients is not None and np.allclose(
        model.coefficients, [[0.5]])
    assert model.initial_values is not None and np.allclose(
        model.initial_values, [[1]])
    assert model.noise_stddev == 0.1
    assert model.output == "show"


def test_generate_var1_data():
    model = sim_VAR()
    model.set_parameters(
        n_obs=10,
        coefficients=[[0.5]],
        initial_values=[[1]],
        noise_stddev=0.0  # No noise for deterministic output
    )

    data = model.generate_var1_data()

    assert data.shape == (10, 1)
    expected_data = np.array(
        [1.0 * 0.5 ** t for t in range(10)]).reshape(10, 1)
    assert np.allclose(data, expected_data)


def test_generate_mvar1_data():
    model = sim_VAR()
    model.set_parameters(
        n_obs=10,
        coefficients=[[0.5, 0.2], [0.3, 0.4]],
        initial_values=[[1], [1]],
        noise_stddev=0.0  # No noise for deterministic output
    )

    coefficientsM = np.array([[0.3, 0.2], [0.1, 0.4]])
    initial_valuesM = np.array([[0.5], [0.5]])

    dataX, dataM = model.generate_mvar1_data(coefficientsM, initial_valuesM)

    assert dataX.shape == (10, 2)
    assert dataM.shape == (10, 2)


def test_simulate_varsim():
    model = sim_VAR()
    model.set_parameters(
        n_obs=10,
        coefficients=[[0.5]],
        initial_values=[[1]],
        noise_stddev=0.0
    )

    model.simulate("VARsim")
    assert model.data is not None and model.data.shape == (10, 1)


def test_simulate_mvarsim():
    model = sim_VAR()
    model.set_parameters(
        n_obs=10,
        coefficients=[[0.5, 0.2], [0.3, 0.4]],
        initial_values=[[1], [1]],
        noise_stddev=0.0,
        coefficientsM=[[0.3, 0.2], [0.1, 0.4]],
        initial_valuesM=[[0.5], [0.5]]
    )

    model.simulate("MVARsim")
    assert model.data is not None and model.data.shape == (10, 2)
    assert model.dataM is not None and model.dataM.shape == (10, 2)


def test_invalid_simulate_command():
    model = sim_VAR()
    with pytest.raises(ValueError, match="Invalid command"):
        model.simulate("INVALID")

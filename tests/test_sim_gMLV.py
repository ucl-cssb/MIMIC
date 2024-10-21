import numpy
from mimic.model_simulate.sim_gMLV import sim_gMLV


def test_gmlv_initialization():
    # Test initialization with default parameters
    model = sim_gMLV(num_species=2, num_metabolites=1)
    assert model.nsp == 2
    assert model.nm == 1
    assert model.mu.shape == (2,)
    assert model.M.shape == (2, 2)
    assert model.beta.shape == (1, 2)
    assert model.epsilon.shape == (2, 0)  # Default perturbations set to 0


def test_gmlv_set_parameters():
    # Test setting specific parameters
    mu = numpy.array([1.0, 0.5])
    M = numpy.array([[-1.0, 0.5], [0.5, -1.0]])
    beta = numpy.array([[0.1, 0.2]])
    epsilon = numpy.array([[0.01, 0.02], [0.03, 0.04]])

    model = sim_gMLV(num_species=2, num_metabolites=1)
    model.set_parameters(mu=mu, M=M, beta=beta, epsilon=epsilon)

    assert numpy.allclose(model.mu, mu)
    assert numpy.allclose(model.M, M)
    assert numpy.allclose(model.beta, beta)
    assert numpy.allclose(model.epsilon, epsilon)


def test_gmlv_simulate_no_metabolites():
    # Test simulation with no metabolites
    mu = numpy.array([1.0, 0.5])
    M = numpy.array([[-1.0, 0.5], [0.5, -1.0]])

    model = sim_gMLV(num_species=2, num_metabolites=0, mu=mu, M=M)
    times = numpy.linspace(0, 10, 11)
    sy0 = numpy.array([1.0, 1.0])  # Initial species concentrations

    yobs, sobs, sy0, mu, M, beta = model.simulate(times, sy0)

    assert yobs.shape == (11, 2)
    assert sobs.shape == (11, 0)  # No metabolites
    assert numpy.allclose(sy0, [1.0, 1.0])
    assert numpy.allclose(mu, [1.0, 0.5])
    assert numpy.allclose(M, [[-1.0, 0.5], [0.5, -1.0]])


def test_gmlv_simulate_with_metabolites():
    # Test simulation with metabolites
    mu = numpy.array([1.0, 0.5])
    M = numpy.array([[-1.0, 0.5], [0.5, -1.0]])
    beta = numpy.array([[0.1, 0.2]])

    model = sim_gMLV(num_species=2, num_metabolites=1, mu=mu, M=M, beta=beta)
    times = numpy.linspace(0, 10, 11)
    # Initial species and metabolite concentrations
    sy0 = numpy.array([1.0, 1.0, 0.0])

    yobs, sobs, sy0, mu, M, beta = model.simulate(times, sy0)

    assert yobs.shape == (11, 2)
    assert sobs.shape == (11, 1)  # One metabolite
    assert numpy.allclose(sy0, [1.0, 1.0, 0.0])
    assert numpy.allclose(mu, [1.0, 0.5])
    assert numpy.allclose(M, [[-1.0, 0.5], [0.5, -1.0]])
    assert numpy.allclose(beta, [[0.1, 0.2]])


def test_gmlv_simulate_with_perturbations():
    # Test simulation with perturbations
    mu = numpy.array([1.0, 0.5])
    M = numpy.array([[-1.0, 0.5], [0.5, -1.0]])
    beta = numpy.array([[0.1, 0.2]])
    epsilon = numpy.array([[0.01, 0.02], [0.03, 0.04]])

    def perturbation_function(t):
        return numpy.array([1.0, 0.5])

    model = sim_gMLV(num_species=2, num_metabolites=1,
                     num_perturbations=2, mu=mu, M=M, beta=beta, epsilon=epsilon)
    times = numpy.linspace(0, 10, 11)
    # Initial species and metabolite concentrations
    sy0 = numpy.array([1.0, 1.0, 0.0])

    yobs, sobs, sy0, mu, M, beta = model.simulate(
        times, sy0, u=perturbation_function)

    assert yobs.shape == (11, 2)
    assert sobs.shape == (11, 1)
    assert numpy.allclose(sy0, [1.0, 1.0, 0.0])
    assert numpy.allclose(mu, [1.0, 0.5])
    assert numpy.allclose(M, [[-1.0, 0.5], [0.5, -1.0]])
    assert numpy.allclose(beta, [[0.1, 0.2]])

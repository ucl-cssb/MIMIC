import numpy
from mimic.model_simulate.sim_gLV import *
# test


def test_glv_1_species_random():
    glv = sim_gLV(num_species=1)
    glv.print_parameters()
    assert glv.nsp == 1
    assert len(glv.mu) == 1
    assert glv.M.shape == (1, 1)


def test_glv_1_species():
    glv = sim_gLV(num_species=1, mu=[1], M=numpy.array([[-1]]))
    glv.print_parameters()
    assert glv.nsp == 1
    assert len(glv.mu) == 1
    assert glv.M.shape == (1, 1)
    assert glv.mu[0] == 1
    assert glv.M[0][0] == -1


def test_glv_1_stationary_species_sim():
    glv = sim_gLV(num_species=1, mu=[0], M=numpy.array([[0]]))
    glv.print_parameters()
    assert glv.nsp == 1
    assert len(glv.mu) == 1
    assert glv.M.shape == (1, 1)
    assert glv.mu[0] == 0
    assert glv.M[0][0] == 0

    times = numpy.linspace(0, 10, 11)
    init_species = numpy.array([1])
    s_obs, init_species, mu, M, epsilon = glv.simulate(times, init_species)
    assert len(s_obs) == 11
    assert len(s_obs[0]) == 1
    assert len(init_species) == 1
    assert len(mu) == 1
    assert len(M) == 1
    assert len(M[0]) == 1
    assert init_species[0] == 1
    assert mu[0] == 0
    assert M[0][0] == 0
    for s in s_obs:
        assert s[0] == 1

    glv = sim_gLV(num_species=1, mu=[1], M=numpy.array([[-1]]))
    glv.print_parameters()
    times = numpy.linspace(0, 10, 11)
    init_species = numpy.array([1])
    s_obs, init_species, mu, M, epsilon = glv.simulate(times, init_species)
    for s in s_obs:
        assert s[0] == 1


def test_glv_1_exponential_species_sim():
    glv = sim_gLV(num_species=1, mu=[1], M=numpy.array([[0]]))
    glv.print_parameters()
    assert glv.nsp == 1
    assert len(glv.mu) == 1
    assert glv.M.shape == (1, 1)
    assert glv.mu[0] == 1
    assert glv.M[0][0] == 0

    times = numpy.linspace(0, 10, 11)
    init_species = numpy.array([1])

    s_obs, init_species, mu, M, epsilon = glv.simulate(times, init_species)
    assert len(s_obs) == 11
    assert len(s_obs[0]) == 1
    assert len(init_species) == 1
    assert len(mu) == 1
    assert len(M) == 1
    assert len(M[0]) == 1
    assert init_species[0] == 1
    assert mu[0] == 1
    assert M[0][0] == 0
    for i, s in enumerate(s_obs):
        assert abs(s[0] - numpy.exp(i)) < 0.01


def test_glv_2_species_random():
    glv = sim_gLV(num_species=2)
    glv.print_parameters()
    assert glv.nsp == 2
    assert len(glv.mu) == 2
    assert glv.M.shape == (2, 2)


def test_glv_2_species():
    glv = sim_gLV(num_species=2, mu=[
        1, 2], M=numpy.array([[-1, 0.5], [0, -2]]))
    glv.print_parameters()
    assert glv.nsp == 2
    assert len(glv.mu) == 2
    assert glv.M.shape == (2, 2)
    assert glv.mu[0] == 1
    assert glv.mu[1] == 2
    assert glv.M[0][0] == -1
    assert glv.M[0][1] == 0.5
    assert glv.M[1][0] == 0
    assert glv.M[1][1] == -2


def test_glv_2_no_interaction_species_sim():
    glv = sim_gLV(num_species=2, mu=[
        1, 2], M=numpy.array([[-1, 0], [0, -1]]))
    glv.print_parameters()
    times = numpy.linspace(0, 100, 11)
    init_species = numpy.array([1, 1])
    s_obs, init_species, mu, M, epsilon = glv.simulate(times, init_species)
    assert s_obs.shape == (11, 2)
    assert abs(s_obs[10][0] - 1) < 0.01
    assert abs(s_obs[10][1] - 2) < 0.01

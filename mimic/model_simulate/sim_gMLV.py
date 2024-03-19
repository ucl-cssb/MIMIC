import random
from typing import List, Optional, Union

import numpy  # do not change this to np, it will break the code
from scipy import stats
from scipy.integrate import odeint

from mimic.model_simulate.base_model import BaseModel


class sim_gMLV(BaseModel):
    def __init__(self, num_species=2, num_metabolites=0, num_perturbations=0, mu=None, M=None, beta=None, epsilon=None):
        super().__init__()
        self.model = "gMLV"

        self.nsp = num_species
        self.nm = num_metabolites
        self.np = num_perturbations

        self.mu = numpy.random.lognormal(
            0.01, 0.5, self.nsp) if mu is None else mu
        if M is None:
            self.M = numpy.zeros((self.nsp, self.nsp))
            # add self repression on the diagonal
            for species_idx in range(self.nsp):
                self.M[species_idx, species_idx] = random.uniform(-0.5, 0.0)

            # add random interactions
            for i in range(self.nsp):
                for j in range(self.nsp):
                    if i == j:
                        continue
                    tau = stats.halfcauchy.rvs(loc=0, scale=0.001)
                    lam = stats.halfcauchy.rvs(loc=0, scale=1)
                    M = stats.norm.rvs(loc=0, scale=tau*lam)
                    # if i == j:
                    #     self.M[i, j] = -abs(M)
                    self.M[i, j] = M
                    # i = random.randint(0, self.nsp-1)
                    # j = random.randint(0, self.nsp-1)
                    # self.M[i, j] = random.normalvariate(mu=0, sigma=0.1)
        else:
            self.M = M

        if beta is None and self.nm > 0:
            self.beta = numpy.zeros((self.nm, self.nsp))
            for _ in range(self.nm):
                i = random.randint(0, self.nm-1)
                j = random.randint(0, self.nsp-1)
                self.beta[i, j] = random.uniform(a=0, b=1)
        else:
            self.beta = beta

        if epsilon is None:
            self.epsilon = numpy.zeros((self.nsp, self.np))

            # add random interactions
            for i in range(self.nsp):
                for j in range(self.np):
                    tau = stats.halfcauchy.rvs(loc=0, scale=1)
                    lam = stats.halfcauchy.rvs(loc=0, scale=1)
                    epsilon = stats.norm.rvs(loc=0, scale=tau * lam)
                    self.epsilon[i, j] = -abs(epsilon)
        else:
            self.epsilon = epsilon

        self.parameters = {"num_species": self.nsp, "num_metabolites": self.nm, "num_perturbations": self.np,
                           "mu": self.mu, "M": self.M, "beta": self.beta, "epsilon": self.epsilon}

    def set_parameters(self, num_species: Optional[int] = None, num_metabolites: Optional[int] = None, num_perturbations: Optional[int] = None,
                       mu: Optional[Union[List[float], numpy.ndarray]] = None, M: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       beta: Optional[Union[List[List[float]], numpy.ndarray]] = None, epsilon: Optional[Union[List[List[float]], numpy.ndarray]] = None):
        """
        Set the parameters for the simulation.

        Parameters:
        num_species (int, optional): The number of species.
        num_metabolites (int, optional): The number of metabolites.
        num_perturbations (int, optional): The number of perturbations.
        mu (list, numpy.ndarray, optional): The specific growth rates.
        M (list, numpy.ndarray, optional): The interaction matrix.
        beta (list, numpy.ndarray, optional): The metabolite production rate matrix.
        epsilon (list, numpy.ndarray, optional): The perturbation matrix.
        """
        if num_species is not None:
            self.nsp = num_species
        if num_metabolites is not None:
            self.nm = num_metabolites
        if num_perturbations is not None:
            self.np = num_perturbations
        if mu is not None:
            self.mu = mu
        if M is not None:
            self.M = M
        if beta is not None:
            self.beta = beta
        if epsilon is not None:
            self.epsilon = epsilon

        self.parameters = {"num_species": self.nsp, "num_metabolites": self.nm, "num_perturbations": self.np,
                           "mu": self.mu, "M": self.M, "beta": self.beta, "epsilon": self.epsilon}

    def simulate(self, times, sy0, u=None):
        self.check_params(self.parameters, 'gMLV')
        syobs = odeint(gMLV, sy0, times, args=(self.nsp, self.np,
                       self.mu, self.M, self.beta, self.epsilon, u))
        yobs = syobs[:, 0:self.nsp]
        sobs = syobs[:, self.nsp:]
        self.data = yobs
        return yobs, sobs, sy0, self.mu, self.M, self.beta


def gMLV(sy, t, nsp, np, mu, M, beta, epsilon, u):
    """
    generalised Lotka Volterra with metabolite production

    :param sy: species + metabolites vector
    :param t: time
    :param nsp: number of species
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :param beta: metabolite production rate matrix
    :param u: a function that returns the perturbation signal at time t
    :return: change in species + metabolites vector
    """

    # separate species and metabolites
    y = sy[:nsp]
    s = sy[nsp:]

    # if np > 0:
    #    for p_i in range(np):
    #        if tp[p_i][0] <= t < tp[p_i][1]:
    #            instantaneous_growth = mu + M @ y + epsilon[:, p_i]
    #       else:
    #            instantaneous_growth = mu + M @ y
    # else:
    #    instantaneous_growth = mu + M @ y

    instantaneous_growth = mu + \
        M @ y if u is None else mu + M @ y + epsilon @ u(t)
    dN = numpy.multiply(y, instantaneous_growth)

    if beta is None:
        dS = []
    else:
        # this is simple production
        # dS = beta @ y

        # metabolite production as in Clark et al., 2021: eqs(4 & 5)
        rho = numpy.dot(beta, y) if len(beta.shape) == 3 else beta
        q = numpy.multiply(rho, instantaneous_growth)
        dS = q @ y

    return numpy.hstack((dN, dS))

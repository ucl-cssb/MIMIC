import random
from typing import List, Optional, Union

import numpy
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import odeint

from mimic.model_simulate.base_model import BaseModel


class sim_gLV(BaseModel):
    """
    Class for a generalised Lotka-Volterra model
    """

    def __init__(self, num_species=2, num_perturbations=0, mu=None, M=None, epsilon=None):
        """
        Constructor for the gLV model

        If no parameters are given for mu and M, a model is randomly generated

        :param num_species: number of species
        :param mu: specific growth rates vector
        :param M: interaction matrix
        """
        # QUESTION: is this not supposed to have num_metabolites?
        super().__init__()
        self.model = "gLV"
        self.nsp = num_species
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
                    self.M[i, j] = M
        else:
            self.M = M

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

        self.parameters = {"num_species": self.nsp,
                           "mu": self.mu, "M": self.M, "epsilon": self.epsilon}

    def set_parameters(self, num_species: Optional[int] = None,
                       mu: Optional[Union[List[float],
                                          NDArray[numpy.float64]]] = None,
                       M: Optional[Union[List[List[float]],
                                         NDArray[numpy.float64]]] = None,
                       epsilon: Optional[Union[List[List[float]], NDArray[numpy
                                                                          .float64]]] = None) -> None:
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            mu (Optional[Union[List[float], numpy.ndarray]]): Growth rates.
            M (Optional[Union[List[List[float]], numpy.ndarray]]): Interaction matrix.
            epsilon (Optional[Union[List[List[float]], numpy.ndarray]]): Perturbation matrix.
        """
        if num_species is not None:
            self.nsp = num_species
        if mu is not None:
            self.mu = mu
        if M is not None:
            self.M = numpy.asarray(M, dtype=numpy.float64)
        if epsilon is not None:
            self.epsilon = numpy.asarray(epsilon, dtype=numpy.float64)

        self.parameters = {"num_species": self.nsp,
                           "mu": self.mu, "M": self.M, "epsilon": self.epsilon}

    # HACK: this is a hack to avoid PyLint's arguments-differ error, but maybe we should change it so that the simulate method in the base class has the same signature as this one
    # pylint: disable=arguments-differ
    def simulate(self, times, init_species, u=None) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Simulate the gLV model.

        Args:
            times (numpy.ndarray): Time points for the simulation.
            init_species (numpy.ndarray): Initial species vector.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple containing:
                - species and metabolites at each time point (numpy.ndarray),
                - initial species and metabolites (numpy.ndarray),
                - specific growth rates (numpy.ndarray),
                - interaction matrix (numpy.ndarray).
        """
        self.check_params(self.parameters, 'gLV')

        s_obs = odeint(glv, init_species, times, args=(
            self.mu, self.M, self.epsilon, u))
        self.data = s_obs
        return s_obs, init_species, self.mu, self.M, self.epsilon


def glv(N, t, mu, M, epsilon, u) -> numpy.ndarray:
    """
    ODE simulation function for generalised Lotka-Volterra model

    :param N: vector of species amounts
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :return: dy/dt
    """
    instantaneous_growth = mu + \
        M @ N if u is None else mu + M @ N + epsilon @ u(t)
    # dN = numpy.multiply(N, instantaneous_growth)
    return numpy.multiply(N, instantaneous_growth)

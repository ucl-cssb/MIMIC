import random
from typing import List, Optional, Union

import numpy  # do not change this to np, it will break the code
from scipy import stats
from scipy.integrate import odeint

from mimic.model_simulate.base_model import BaseModel


class sim_gMLV(BaseModel):
    """
    Simulation class for generalized Lotka-Volterra (gMLV) models with metabolite production.

    This class extends the BaseModel to support simulations of ecological systems or microbial communities
    described by the gMLV model, which can include interactions between species, production of metabolites,
    and responses to external perturbations.

    Attributes:
        num_species (int): The number of species in the simulation.
        num_metabolites (int): The number of metabolites produced by the species.
        num_perturbations (int): The number of external perturbations applied to the system.
        mu (numpy.ndarray): Vector of intrinsic growth rates of the species.
        M (numpy.ndarray): Interaction matrix describing the effects of species on each other.
        beta (numpy.ndarray): Metabolite production matrix describing the production rates of metabolites by species.
        epsilon (numpy.ndarray): Perturbation matrix describing the effects of external perturbations on species.

    Methods:
        set_parameters: Sets or updates the parameters for the simulation.
        simulate: Runs the gMLV simulation over a specified time course and initial conditions.
    """

    def __init__(
            self,
            num_species=2,
            num_metabolites=0,
            num_perturbations=0,
            mu=None,
            M=None,
            beta=None,
            epsilon=None):
        """
        Initializes the gMLV simulation with given parameters or defaults.

        Parameters:
            num_species (int, optional): The number of species. Defaults to 2.
            num_metabolites (int, optional): The number of metabolites. Defaults to 0.
            num_perturbations (int, optional): The number of perturbations. Defaults to 0.
            mu (numpy.ndarray, optional): Intrinsic growth rates. Random lognormal values by default.
            M (numpy.ndarray, optional): Interaction matrix. Random values with self-repression on diagonal by default.
            beta (numpy.ndarray, optional): Metabolite production rates. Zeros by default or random values if num_metabolites > 0.
            epsilon (numpy.ndarray, optional): Perturbation effects. Zeros by default.
        """
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
                    M = stats.norm.rvs(loc=0, scale=tau * lam)
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
                i = random.randint(0, self.nm - 1)
                j = random.randint(0, self.nsp - 1)
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

        self.parameters = {
            "num_species": self.nsp,
            "num_metabolites": self.nm,
            "num_perturbations": self.np,
            "mu": self.mu,
            "M": self.M,
            "beta": self.beta,
            "epsilon": self.epsilon}

    def set_parameters(self,
                       num_species: Optional[int] = None,
                       num_metabolites: Optional[int] = None,
                       num_perturbations: Optional[int] = None,
                       mu: Optional[Union[List[float],
                                          numpy.ndarray]] = None,
                       M: Optional[Union[List[List[float]],
                                         numpy.ndarray]] = None,
                       beta: Optional[Union[List[List[float]],
                                            numpy.ndarray]] = None,
                       epsilon: Optional[Union[List[List[float]],
                                               numpy.ndarray]] = None) -> None:
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            num_metabolites (Optional[int]): Number of metabolites.
            num_perturbations (Optional[int]): Number of perturbations.
            mu (Optional[Union[List[float], numpy.ndarray]]): Growth rates.
            M (Optional[Union[List[List[float]], numpy.ndarray]]): Interaction matrix.
            beta (Optional[Union[List[List[float]], numpy.ndarray]]): Metabolite production rates.
            epsilon (Optional[Union[List[List[float]], numpy.ndarray]]): Perturbation effects.
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
            self.M = numpy.asarray(M, dtype=numpy.float64)
        if beta is not None:
            self.beta = numpy.asarray(beta, dtype=numpy.float64)
        if epsilon is not None:
            self.epsilon = numpy.asarray(epsilon, dtype=numpy.float64)

        self.parameters = {
            "num_species": self.nsp,
            "num_metabolites": self.nm,
            "num_perturbations": self.np,
            "mu": self.mu,
            "M": self.M,
            "beta": self.beta,
            "epsilon": self.epsilon}

    # HACK: this is a hack to avoid PyLint's arguments-differ error, but maybe we should change it so that the simulate method in the base class has the same signature as this one
    # pylint: disable=arguments-differ

    def simulate(self,
                 times,
                 sy0,
                 u=None) -> tuple[numpy.ndarray,
                                  numpy.ndarray,
                                  numpy.ndarray,
                                  numpy.ndarray,
                                  numpy.ndarray,
                                  numpy.ndarray]:
        """
        Runs the gMLV simulation over the specified time course with given initial conditions and optional perturbations.

        Parameters:
            times (numpy.ndarray): Array of time points at which to simulate.
            sy0 (numpy.ndarray): Initial conditions for species and metabolites.
            u (callable, optional): Function representing the external perturbation signal over time.

        Returns:
            tuple: Tuple containing the simulation results for species (yobs), metabolites (sobs),
            initial conditions (sy0), growth rates (mu), interaction matrix (M), and metabolite production rates (beta).
        """
        self.check_params(self.parameters, 'gMLV')
        syobs = odeint(gMLV, sy0, times, args=(self.nsp, self.np,
                       self.mu, self.M, self.beta, self.epsilon, u))
        yobs = syobs[:, 0:self.nsp]
        sobs = syobs[:, self.nsp:]
        self.data = yobs  # QUESTION: should this be yobs or sobs?
        return yobs, sobs, sy0, self.mu, self.M, self.beta


def gMLV(sy, t, nsp, np, mu, M, beta, epsilon, u) -> numpy.ndarray:
    """
    Differential equations for the generalized Lotka-Volterra model with metabolite production.

    Parameters:
        sy (numpy.ndarray): Combined vector of species and metabolites at the current time.
        t (float): Current time point.
        nsp (int): Number of species.
        np (int): Number of perturbations.
        mu (numpy.ndarray): Specific growth rates vector.
        M (numpy.ndarray): Interaction matrix.
        beta (numpy.ndarray): Metabolite production rate matrix.
        epsilon (numpy.ndarray): Perturbation matrix.
        u (callable): Function to compute the perturbation signal at time t.

    Returns:
        numpy.ndarray: The derivative of the combined species and metabolites vector.
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

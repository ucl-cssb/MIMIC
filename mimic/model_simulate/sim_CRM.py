import random
from typing import List, Optional, Union

import numpy  # do not change this to np, it will break the code
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import odeint

from mimic.model_simulate.base_model import BaseModel


class sim_CRM(BaseModel):
    """
    Simulation class for Consumer-Resource Model (CRM).

    This class extends the BaseModel to support simulations of ecological systems or microbial communities described by
    the MacArthur CRM, which can include a community of consumer species competing for a common pool of resources.

    Attributes:
        num_species (int): The number of species in the simulation.
        num_resources (int): The number of resources in the simulation.

    Methods:
        set_parameters: Sets or updates the parameters for the simulation.
        simulate: Runs the CRM simulation over a specified time course and initial conditions.
    """

    def __init__(
            self,
            num_species=2,
            num_resources=0):
        """
        Initializes the CRM simulation with given parameters or defaults.

        Parameters:
            num_species (int): The number of species. Defaults to 2.
            num_resources (int): The number of resources. Defaults to 0.
        """
        super().__init__()
        self.model = "CRM"

        self.nsp = num_species
        self.nr = num_resources

        self.tau = numpy.ones(num_species),  # species_timescale
        self.r = numpy.ones(num_resources),  # resource_timescale

        self.w = numpy.ones(num_resources),  # resource_quality
        self.c = numpy.ones((num_species, num_resources)
                            ),  # relative_preference
        self.m = numpy.ones(num_species),  # mortality_rate

        self.K = numpy.ones(num_resources)  # resource_capacity

        self.parameters = {
            "num_species": self.nsp,
            "num_resources": self.nr,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K}

    def set_parameters(self,
                       num_species: Optional[int] = None,
                       num_resources: Optional[int] = None,
                       tau: Optional[Union[List[float], numpy.ndarray]] = None,
                       w: Optional[Union[List[float], numpy.ndarray]] = None,
                       c: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       m: Optional[Union[List[float], numpy.ndarray]] = None,
                       r: Optional[Union[List[float], numpy.ndarray]] = None,
                       K: Optional[Union[List[float], numpy.ndarray]] = None) -> None:
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            num_resources (Optional[int]): Number of resources.
            tau (Optional[Union[List[float], numpy.ndarray]]): species_timescales.
            w (Optional[Union[List[float], numpy.ndarray]]): resource_quality.
            c (Optional[Union[List[List[float]], numpy.ndarray]]): relative_preference.
            m (Optional[Union[List[float], numpy.ndarray]]): mortality_rate.
            r (Optional[Union[List[float], numpy.ndarray]]): resource_timescale.
            K (Optional[Union[List[float], numpy.ndarray]]): resource_capacity.
        """
        if num_species is not None:
            self.nsp = num_species
        if num_resources is not None:
            self.nr = num_resources
        if tau is not None:
            self.tau = tau
        if w is not None:
            self.w = w
        if c is not None:
            self.c = numpy.asarray(c, dtype=numpy.float64)
        if m is not None:
            self.m = m
        if r is not None:
            self.r = r
        if K is not None:
            self.K = K

        self.parameters = {
            "num_species": self.nsp,
            "num_resources": self.nr,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K}

    # HACK: this is a hack to avoid PyLint's arguments-differ error, but maybe we should change it so that the simulate method in the base class has the same signature as this one
    # pylint: disable=arguments-differ

    def simulate(self,
                 times,
                 sy0) -> tuple[numpy.ndarray,
                               numpy.ndarray]:
        """
        Runs the CRM simulation over the specified time course with given initial conditions.

        Parameters:
            times (numpy.ndarray): Array of time points at which to simulate.
            sy0 (numpy.ndarray): Initial conditions for species and resources.

        Returns:
            tuple: Tuple containing the simulation results for species (yobs), metabolites (sobs),
            initial conditions (sy0)
        """
        # self.check_params(self.parameters, 'CRM')
        syobs = odeint(
            CRM,
            sy0,
            times,
            args=(
                self.nsp,
                self.nr,
                self.tau,
                self.w,
                self.c,
                self.m,
                self.r,
                self.K))
        yobs = syobs[:, 0:self.nsp]  # species
        sobs = syobs[:, self.nsp:]  # resources
        self.data = syobs 
        return yobs, sobs


def CRM(sy, t, nsp, nr, tau, w, c, m, r, K) -> numpy.ndarray:
    """
    Differential equations for the CRM.

    Parameters:
        sy (numpy.ndarray): Combined vector of species and resources at the current time.
        t (float): Current time point.
        nsp (int): Number of species.
        nr (int): Number of resources.
        tau (numpy.ndarray): Vector of species timescales.
        w (numpy.ndarray): Vector of resource qualities.
        c (numpy.ndarray): Matrix of species preferences for resources.
        m (numpy.ndarray): Vector of species mortality rates.
        r (numpy.ndarray): Vector of resource timescales.
        K (numpy.ndarray): Vector of resource capacities.

    Returns:
        numpy.ndarray: The derivative of the combined species and resources vector.
    """

    # separate species and metabolites
    N = sy[:nsp]
    R = sy[nsp:]

    # dN_i/dt = 1/tau_i * N_i * (Sum_a(c_ia * w_a * R_a - m_i))
    dN = (N / tau) * (c @ (w * R) - m)

    # dR_a/dt = 1/(r_a * K_a) * (K_a - R_a) * R_a - Sum_i(N_i * c_ia * R_a)
    dR = (1 / r * K) * (K - R) * R - (N @ c * R)

    return numpy.hstack((dN, dR))

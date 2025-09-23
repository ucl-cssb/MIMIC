import random
from typing import List, Optional, Union

import numpy  # do not change this to np, it will break the code
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import odeint

from mimic.model_simulate.base_model import BaseModel


class sim_Gompertz(BaseModel):
    """
    Simulation class for Gompertz growth model.

    This class extends the BaseModel to support simulations of Gompertz growth curves
    for multiple species with no interaction between them.

    Each species follows: dN_i/dt = mu_i * N_i * ln(A_i/N_i)

    Attributes:
        num_species (int): The number of species in the simulation.

    Methods:
        set_parameters: Sets or updates the parameters for the simulation.
        simulate: Runs the Gompertz simulation over a specified time course and initial conditions.
    """

    def __init__(
            self,
            num_species=2,
            mu=None,
            A=None):
        """
        Initializes the Gompertz simulation with given parameters or defaults.

        Parameters:
            num_species (int): The number of species. Defaults to 2.
            mu (Optional[Union[List[float], numpy.ndarray]]): Growth rate parameters.
            A (Optional[Union[List[float], numpy.ndarray]]): Carry capacity values.
        """
        super().__init__()
        self.model = "Gompertz"

        self.nsp = num_species

        # Default parameter values
        self.mu =  numpy.ones(num_species)  # Growth rate parameters
        self.A = numpy.full(num_species, 10.0)  # Carry capacity parameters

        self.parameters = {
            "num_species": self.nsp,
            "mu": self.mu,
            "A": self.A
        }

    def set_parameters(self,
                       num_species: Optional[int] = None,
                       mu: Optional[Union[List[float], numpy.ndarray]] = None,
                       A: Optional[Union[List[float], numpy.ndarray]] = None) -> None:
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            mu (Optional[Union[List[float], numpy.ndarray]]): Growth rate parameters.
            A (Optional[Union[List[float], numpy.ndarray]]): Asymptotic maximum values.
        """
        if num_species is not None:
            self.nsp = num_species
        if mu is not None:
            self.mu = numpy.asarray(mu, dtype=numpy.float64)
        if A is not None:
            self.A = numpy.asarray(A, dtype=numpy.float64)

        self.parameters = {
            "num_species": self.nsp,
            "mu": self.mu,
            "A": self.A
        }

    def simulate(self,
                 times,
                 y0) -> tuple[numpy.ndarray]:
        """
        Runs the Gompertz simulation over the specified time course with given initial conditions.

        Parameters:
            times (numpy.ndarray): Array of time points at which to simulate.
            y0 (numpy.ndarray): Initial conditions for species populations.

        Returns:
            tuple: Tuple containing the simulation results for species (yobs)
        """
        yobs = odeint(
            Gompertz,
            y0,
            times,
            args=(
                self.nsp,
                self.mu,
                self.A))
        
        self.data = yobs
        return yobs,  # Return as tuple for consistency with CRM structure


def Gompertz(y, t, nsp, mu, A) -> numpy.ndarray:
    """
    Differential equations for Gompertz growth.

    Each species grows independently according to:
    dN_i/dt = mu_i * N_i * ln(A_i/N_i)

    Parameters:
        y (numpy.ndarray): Vector of species populations at the current time.
        t (float): Current time point.
        nsp (int): Number of species.
        mu (numpy.ndarray): Vector of growth rate parameters.
        A (numpy.ndarray): Vector of carry capacity values.

    Returns:
        numpy.ndarray: The derivative of the species population vector.
    """

    N = y[:nsp]  # Species populations

    # Ensure positive populations and avoid division issues
    eps = 1e-8
    N_safe = numpy.maximum(N, eps)
    A_safe = numpy.maximum(A, eps)
    
    # Ensure N doesn't exceed A to avoid negative ln
    N_bounded = numpy.minimum(N_safe, A_safe - eps)

    # Independent Gompertz growth equations
    # dN_i/dt = mu_i * N_i * ln(A_i/N_i)
    ln_term = numpy.log(A_safe / N_bounded)
    dN = mu * N_bounded * ln_term

    # Prevent negative derivatives when population is very small
    dN = numpy.where((N < eps) & (dN < 0), 0.0, dN)

    return dN
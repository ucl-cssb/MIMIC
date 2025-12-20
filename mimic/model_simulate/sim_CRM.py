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
            num_resources=2,
            num_secondary_resources=0,
            tau=None,
            w=None,
            c=None,
            m=None,
            r=None,
            K=None,
            alpha=None,
            epsilon=None):
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
        self.nsec = num_secondary_resources

        self.tau = numpy.ones(num_species)  # species_timescale
        self.r = numpy.ones(num_resources)  # resource_timescale

        self.w = numpy.ones(num_resources+num_secondary_resources)  # resource_quality
        self.c = numpy.ones((num_species, num_resources+num_secondary_resources))  # relative_preference
        self.m = numpy.ones(num_species)  # mortality_rate

        self.K = numpy.ones(num_resources)  # resource_capacity
        self.alpha = numpy.ones((num_species, num_resources))  # NH4+ production rate
        self.epsilon = numpy.ones((num_species, num_resources))  # resource recycling efficiency from mortality

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
                       num_secondary_resources: Optional[int] = None,
                       tau: Optional[Union[List[float], numpy.ndarray]] = None,
                       w: Optional[Union[List[float], numpy.ndarray]] = None,
                       c: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       m: Optional[Union[List[float], numpy.ndarray]] = None,
                       r: Optional[Union[List[float], numpy.ndarray]] = None,
                       K: Optional[Union[List[float], numpy.ndarray]] = None,
                       alpha: Optional[Union[List[List[float]], numpy.ndarray]] = None,
                       epsilon: Optional[Union[List[List[float]], numpy.ndarray]] = None):
        """
        Updates the simulation parameters. Only provided values are updated; others remain unchanged.

        Parameters:
            num_species (Optional[int]): Number of species.
            num_resources (Optional[int]): Number of resources.
            num_secondary_resources (Optional[int]): Number of secondary resources.
            tau (Optional[Union[List[float], numpy.ndarray]]): species_timescales.
            w (Optional[Union[List[float], numpy.ndarray]]): resource_quality.
            c (Optional[Union[List[List[float]], numpy.ndarray]]): relative_preference.
            m (Optional[Union[List[float], numpy.ndarray]]): mortality_rate.
            r (Optional[Union[List[float], numpy.ndarray]]): resource_timescale.
            K (Optional[Union[List[float], numpy.ndarray]]): resource_capacity.
            alpha (Optional[Union[List[List[float]], numpy.ndarray]]): NH4+ production rate.
            epsilon (Optional[Union[List[List[float]], numpy.ndarray]]): resource recycling efficiency from mortality.
        """
        if num_species is not None:
            self.nsp = num_species
        if num_resources is not None:
            self.nr = num_resources
        if num_secondary_resources is not None:
            self.nsec = num_secondary_resources
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
        if alpha is not None:
            self.alpha = numpy.asarray(alpha, dtype=numpy.float64)
        if epsilon is not None:
            self.epsilon = numpy.asarray(epsilon, dtype=numpy.float64) 

        self.parameters = {
            "num_species": self.nsp,
            "num_resources": self.nr,
            "num_secondary_resources": self.nsec,
            "tau": self.tau,
            "w": self.w,
            "c": self.c,
            "m": self.m,
            "r": self.r,
            "K": self.K,
            "alpha": self.alpha,
            "epsilon": self.epsilon}

    # HACK: this is a hack to avoid PyLint's arguments-differ error, but maybe we should change it so that the simulate method in the base class has the same signature as this one
    # pylint: disable=arguments-differ

    def simulate(self, times, sy0, model_type='Rep'):
        """
        Runs the CRM simulation over the specified time course with given initial conditions.
        
        Parameters:
            times (numpy.ndarray): Array of time points at which to simulate.
            sy0 (numpy.ndarray): Initial conditions for species and resources.
            model_type (str): Type of CRM model to simulate. Options: 'Rep', 'NonRep', 'CoLim', 'CoLim_MortRec'
        
        Returns:
            tuple: Tuple containing the simulation results for species (yobs), resources (robs/sobs),
                and initial conditions (sy0)
        """
        if model_type == 'Rep':
            return self.simulate_Rep_CRM(times, sy0)
        elif model_type == 'NonRep':
            return self.simulate_NonRep_CRM(times, sy0)
        elif model_type == 'CoLim':
            return self.simulate_CoLim_CRM(times, sy0)
        elif model_type == 'CoLim_MortRec':
            return self.simulate_CoLim_MortRec_CRM(times, sy0)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from 'Rep', 'NonRep', 'CoLim', 'CoLim_MortRec'")

    def simulate_Rep_CRM(self,
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
            Rep_CRM,
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
        robs = syobs[:, self.nsp:]  # resources
        self.data = syobs
        return yobs, robs
    
    def simulate_NonRep_CRM(self,
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
            NonRep_CRM,
            sy0,
            times,
            args=(
                self.nsp,
                self.nr,
                self.tau,
                self.w,
                self.c,
                self.m))
        yobs = syobs[:, 0:self.nsp]  # species
        robs = syobs[:, self.nsp:]  # resources
        self.data = syobs
        return yobs, robs
    

    def simulate_CoLim_CRM(self,
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
            CoLim_CRM,
            sy0,
            times,
            args=(
                self.nsp,
                self.nr,
                self.nsec,
                self.tau,
                self.w,
                self.c,
                self.m,
                self.alpha))
        yobs = syobs[:, 0:self.nsp]  # species
        robs = syobs[:, self.nsp:self.nsp + self.nr]  # carbon resources
        sobs = syobs[:, self.nsp + self.nr:]  # secondary resources

        self.data = syobs
        return yobs, robs, sobs
    
    def simulate_CoLim_MortRec_CRM(self,
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
            CoLim_MortRec_CRM,
            sy0,
            times,
            args=(
                self.nsp,
                self.nr,
                self.nsec,
                self.tau,
                self.w,
                self.c,
                self.m,
                self.alpha,
                self.epsilon))
        yobs = syobs[:, 0:self.nsp]  # species
        robs = syobs[:, self.nsp:self.nsp + self.nr]  # carbon resources
        sobs = syobs[:, self.nsp + self.nr:]  # secondary resources

        self.data = syobs
        return yobs, robs, sobs


def Rep_CRM(sy, t, nsp, nr, tau, w, c, m, r, K) -> numpy.ndarray:
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
    dR = (r / K) * (K - R) * R - (N @ c * R)

    return numpy.hstack((dN, dR))


def NonRep_CRM(sy, t, nsp, nr, tau, w, c, m) -> numpy.ndarray:
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

    Returns:
        numpy.ndarray: The derivative of the combined species and resources vector.
    """

    # separate species and metabolites
    N = sy[:nsp]
    R = sy[nsp:]

    # dN_i/dt = 1/tau_i * N_i * (Sum_a(c_ia * w_a * R_a - m_i))
    dN = (N / tau) * (c @ (w * R) - m)

    # dR_a/dt = 1/(r_a * K_a) * (K_a - R_a) * R_a - Sum_i(N_i * c_ia * R_a)
    dR =  - (N @ c * R)

    return numpy.hstack((dN, dR))


def CoLim_CRM(sy, t, nsp, nr, nsec, tau, w, c, m, alpha) -> numpy.ndarray:
    """
    Differential equations for the Co-Limiting CRM.
    
    Parameters:
        sy (numpy.ndarray): Combined vector of species, carbon resources, and secondary resources.
        t (float): Current time point.
        nsp (int): Number of species.
        nr (int): Number of carbon resources.
        nsec (int): Number of secondary resources (e.g., NH4+).
        tau (numpy.ndarray): Vector of species timescales (shape: nsp).
        w (numpy.ndarray): Vector of resource qualities (shape: nr+nsec).
        c (numpy.ndarray): Matrix of species preferences for all resources (shape: nsp x (nr+nsec)).
        m (numpy.ndarray): Vector of species mortality rates (shape: nsp).
        alpha (numpy.ndarray): Matrix of NH4+ production rates (shape: nsp x nr).
    
    Returns:
        numpy.ndarray: The derivative of the combined species and resources vector.
    """
    # Separate species and resources
    N = sy[:nsp]
    R = sy[nsp:nsp + nr]
    S = sy[nsp + nr:]
    
    # Split c and w into carbon and secondary resources
    w_carbon = w[:nr]
    w_sec = w[nr:]

    c_carbon = c[:, :nr]
    c_sec = c[:, nr:]
    
    # dN_i/dt = (N_i/τ_i)[c_ik w_k S_k × (Σ_j c_ij w_j R_j) - m_i]
    dN = (N / tau) * ((c_sec @ (w_sec * S)) * (c_carbon @ (w_carbon * R)) - m)
    # dR_j/dt = -Σ_i N_i c_ij R_j
    dR = -((N @ c_carbon) * R)
    # dS_k/dt = -Σ_i N_i c_ik S_k + Σ_i Σ_j α_ij N_i c_ij R_j
    dS = -(N @ c_sec) * S + numpy.sum(alpha * N[:, None] * c_carbon * R[None, :])
    
    return numpy.hstack((dN, dR, dS))


def CoLim_MortRec_CRM(sy, t, nsp, nr, nsec, tau, w, c, m, alpha, epsilon) -> numpy.ndarray:
    """
    Differential equations for the Co-Limiting Mortality-Recycling CRM.
    
    Parameters:
        sy (numpy.ndarray): Combined vector of species, carbon resources, and secondary resources.
        t (float): Current time point.
        nsp (int): Number of species.
        nr (int): Number of carbon resources.
        nsec (int): Number of secondary resources (e.g., NH4+).
        tau (numpy.ndarray): Vector of species timescales (shape: nsp).
        w_carbon (numpy.ndarray): Vector of carbon resource qualities (shape: nr).
        w_sec (numpy.ndarray): Vector of secondary resource qualities (shape: nsec).
        c (numpy.ndarray): Matrix of species preferences for all resources (shape: nsp x (nr+nsec)).
        m (numpy.ndarray): Vector of species mortality rates (shape: nsp).
        alpha (numpy.ndarray): Matrix of NH4+ production rates (shape: nsp x nr).
        epsilon (numpy.ndarray): Matrix of resource recycling efficiency from mortality (shape: nsp x nr).
    
    Returns:
        numpy.ndarray: The derivative of the combined species and resources vector.
    """
    # Separate species and resources
    N = sy[:nsp]
    R = sy[nsp:nsp + nr]
    S = sy[nsp + nr:]
    
    # Split c matrix
    c_carbon = c[:, :nr]
    c_sec = c[:, nr:]

    w_carbon = w[:nr]
    w_sec = w[nr:]
    
    # dN_i/dt = (N_i/τ_i)[c_ik w_k S_k × (Σ_j c_ij w_j R_j) - m_i]
    dN = (N / tau) * ((c_sec @ (w_sec * S)) * (c_carbon @ (w_carbon * R)) - m)
    # dR_j/dt = -Σ_i N_i c_ij R_j + Σ_i ε_ij m_i N_i
    dR = -((N @ c_carbon) * R) + ((m * N) @ epsilon)
    # dS_k/dt = -Σ_i N_i c_ik S_k + Σ_i Σ_j α_ij N_i c_ij R_j
    dS = -((N @ c_sec) * S) + numpy.sum(alpha * N[:, None] * c_carbon * R[None, :])
    
    return numpy.hstack((dN, dR, dS))

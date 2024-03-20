import random
from scipy import stats
import numpy
from scipy.integrate import odeint


class GlvSim:
    """
    Class for a generalised Lotka-Volterra model
    """

    def __init__(self, num_species=2, mu=None, M=None):
        """
        Constructor for the gLV model

        If no parameters are given for mu and M, a model is randomly generated

        :param num_species: number of species
        :param mu: specific growth rates vector
        :param M: interaction matrix
        """

        self.nsp = num_species

        if mu is None:
            # randomly generate specific growth rates
            self.mu = numpy.random.lognormal(0.01, 0.5, self.nsp)
        else:
            self.mu = mu

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
                    else:
                        tau = stats.halfcauchy.rvs(loc=0, scale=0.001)
                        lam = stats.halfcauchy.rvs(loc=0, scale=1)
                        M = stats.norm.rvs(loc=0, scale=tau*lam)
                        self.M[i, j] = M
        else:
            self.M = M

    def simulate(self, times, init_species):
        """
        Simulate the gLV model

        :param times: time points
        :param init_species: initial species vector
        :return: species and metabolites at each time point, initial species and metabolites, specific growth rates,
        interaction matrix, metabolite production rate matrix, perturbation matrix
        """

        s_obs = odeint(glv, init_species, times, args=(self.mu, self.M))

        return s_obs, init_species, self.mu, self.M

    def print(self):
        """
        Print the gLV model
        """
        print(f'number of species: {self.nsp}')
        print(f'specific growth rates: {self.mu}')
        print(f'interaction matrix: \n{self.M}')


def glv(N, t, mu, M):
    """
    ODE simulation function for generalised Lotka-Volterra model

    :param N: vector of species amounts
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :return: dy/dt
    """

    # dN/dt = mu * N + N * M * N
    dN = numpy.multiply(mu, N) + numpy.multiply(N, M @ N)
    return dN

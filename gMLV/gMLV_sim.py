import random
import numpy as np
from scipy.integrate import odeint


class gMLV_sim:
    def __init__(self, num_species=2, num_metabolites=0, mu=None, M=None, beta=None, epsilon=None, C=None):
        self.nsp = num_species
        self.nm = num_metabolites

        if mu is None:
            self.mu = np.random.lognormal(0.01, 0.5, self.nsp)
        else:
            self.mu = mu

        if M is None:
            self.M = np.zeros((self.nsp, self.nsp))
            # add self repression on the diagonal
            for species_idx in range(self.nsp):
                self.M[species_idx, species_idx] = random.uniform(-0.5, 0.0)

            # add random interactions
            for _ in range(self.nsp):
                i = random.randint(0, self.nsp-1)
                j = random.randint(0, self.nsp-1)
                self.M[i, j] = random.normalvariate(mu=0, sigma=0.1)
        else:
            self.M = M

        if beta is None and self.nm > 0:
            self.beta = np.zeros((self.nm, self.nsp))
            for _ in range(self.nm):
                i = random.randint(0, self.nm-1)
                j = random.randint(0, self.nsp-1)
                self.beta[i, j] = random.uniform(a=0, b=1)
        else:
            self.beta = beta

        self.epsilon = epsilon
        self.C = C

    def simulate(self, times, sy0, p=None):
        syobs = odeint(gMLV, sy0, times, args=(self.nsp, self.mu, self.M, self.beta, self.C, p))
        yobs = syobs[:, 0:self.nsp]
        sobs = syobs[:, self.nsp:]
        return yobs, sobs, sy0, self.mu, self.M, self.beta

    def print(self):
        print(f'number of species: {self.nsp}')
        print(f'specific growth rates: {self.mu}')
        print(f'interaction matrix: \n{self.M}')
        print(f'metabolite production: \n{self.beta}')


def gMLV(sy, t, nsp, mu, M, beta, C, p):
    """
    generalised Lotka Volterra with metabolite production

    :param sy: species + metabolites vector
    :param t: time
    :param nsp: number of species
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :param beta: metabolite production rate matrix
    :param p: perturbation function that returns the perturbation vector as a function of time
    :return: change in species + metabolites vector
    """

    # separate species and metabolites

    sy[sy < 0] = 0
    y = sy[0:nsp]
    s = sy[nsp:]



    if p is None:
        instantaneous_growth = mu + M @ y
        # dN = np.multiply(mu, y) + np.multiply(y, M @ y)
    else:

        #instantaneous_growth = mu + M @ y + p[1]

        instantaneous_growth = mu + M @ y + C @ p(t)
        # dN = np.multiply(mu, y) + np.multiply(y, M @ y) + np.multiply(y, p[1])

    dN = np.multiply(y, instantaneous_growth)

    if beta is None:
        dS = []
    else:
        # this is simple production
        # dS = beta @ y

        # metabolite production as in Clark et al., 2021: eqs(4 & 5)
        if len(beta.shape) == 3:
            rho = np.dot(beta, y)  # eq(6)
        else:
            rho = beta
        q = np.multiply(rho, instantaneous_growth)
        dS = q @ y

    return np.hstack((dN, dS))

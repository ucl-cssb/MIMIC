import random
import numpy as np
from scipy.integrate import odeint


class gMLV_sim:
    def __init__(self, num_species=2, num_metabolites=0, mu=None, M=None, beta=None):
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

    def simulate(self, times, sy0):
        syobs = odeint(gMLV, sy0, times, args=(self.nsp, self.mu, self.M, self.beta))
        yobs = syobs[:, 0:self.nsp]
        sobs = syobs[:, self.nsp:]
        return yobs, sobs, sy0, self.mu, self.M, self.beta

    def print(self):
        print(f'number of species: {self.nsp}')
        print(f'specific growth rates: {self.mu}')
        print(f'interaction matrix: \n{self.M}')
        print(f'metabolite production: \n{self.beta}')


def gMLV(sy, t, nsp, mu, M, beta):
    """
    generalised Lotka Volterra with metabolite production

    :param sy: species + metabolites vector
    :param t: time
    :param nsp: number of species
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :param beta: metabolite production rate matrix
    :return: change in species + metabolites vector
    """

    # separate species and metabolites
    y = sy[0:nsp]
    s = sy[nsp:]

    dN = np.multiply(mu, y) + np.multiply(y, M @ y)

    if beta is None:
        dS = []
    else:
        # this is simple production
        dS = beta @ y
    
        # this is growth linked production: model need reconsidering
        # dS = beta @ mp.multiply( dN, y )
    
    return np.hstack((dN, dS))

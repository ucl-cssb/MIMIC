import random
from scipy import stats
import numpy
from scipy.integrate import odeint


class gMLV_sim:
    def __init__(self, num_species=2, num_metabolites=0, num_perturbations=0, mu=None, M=None, beta=None, epsilon=None):
        self.nsp = num_species
        self.nm = num_metabolites
        self.np = num_perturbations

        if mu is None:
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

    def simulate(self, times, sy0, tp=None):
        syobs = odeint(gMLV, sy0, times, args=(self.nsp, self.np, self.mu, self.M, self.beta, self.epsilon, tp))
        yobs = syobs[:, 0:self.nsp]
        sobs = syobs[:, self.nsp:]
        return yobs, sobs, sy0, self.mu, self.M, self.beta

    def print(self):
        print(f'number of species: {self.nsp}')
        print(f'specific growth rates: {self.mu}')
        print(f'interaction matrix: \n{self.M}')
        print(f'metabolite production: \n{self.beta}')
        print(f'perturbation matrix: \n{self.epsilon}')


def gMLV(sy, t, nsp, np, mu, M, beta, epsilon, tp):
    """
    generalised Lotka Volterra with metabolite production

    :param sy: species + metabolites vector
    :param t: time
    :param nsp: number of species
    :param mu: specific growth rates vector
    :param M: interaction matrix
    :param beta: metabolite production rate matrix
    :param p: a tuple containing time-dependent perturbation and perturbation matrix
    :return: change in species + metabolites vector
    """
    
    # separate species and metabolites
    y = sy[0:nsp]
    s = sy[nsp:]

    if np > 0:
        for p_i in range(np):
            if tp[p_i][0] <= t < tp[p_i][1]:
                instantaneous_growth = mu + M @ y + epsilon[:, p_i]
            else:
                instantaneous_growth = mu + M @ y
    else:
        instantaneous_growth = mu + M @ y

    # if p[0] is None:
    #     instantaneous_growth = mu + M @ y
    #     # dN = np.multiply(mu, y) + np.multiply(y, M @ y)
    # else:
    #     if p[0] <= t < (p[0] + 1):
    #         instantaneous_growth = mu + M @ y + p[1]
    #         # dN = np.multiply(mu, y) + np.multiply(y, M @ y) + np.multiply(y, p[1])
    #     else:
    #         instantaneous_growth = mu + M @ y
    #         # dN = np.multiply(mu, y) + np.multiply(y, M @ y)
    dN = numpy.multiply(y, instantaneous_growth)

    if beta is None:
        dS = []
    else:
        # this is simple production
        # dS = beta @ y

        # metabolite production as in Clark et al., 2021: eqs(4 & 5)
        if len(beta.shape) == 3:
            rho = numpy.dot(beta, y)  # eq(6)
        else:
            rho = beta
        q = numpy.multiply(rho, instantaneous_growth)
        dS = q @ y

    return numpy.hstack((dN, dS))

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



def generate_params(num_species, num_pert, zero_prop = 0, hetergeneous = False):

    '''
    generates parameters for GLV simulation according to Cao et al 2017
     (Inferring human microbial dynamics from temporal metagenomics data: Pitfalls and lessons)
     Method in the supplimentary
     num_species: number of microbial strains
     num_perterbations: number of perterbations
     zero_prop: proportion of the interaction matrix that should be zeros
    '''

    N = np.random.normal(0, 1, (num_species, num_species))

    if hetergeneous:
        y = 1.2
        u = np.random.uniform(0, 1, size=(num_species))
        H = (1-u)**(1/(1-y))
        H = np.diag(H)
        s = np.sum(H)
    else:
        H = np.eye(num_species)
        #s = 3 #from the paper
        s = np.sum(H) # this seems to prevent instability when more species

    a = np.random.binomial(1, 1-zero_prop, size=(num_species, num_species))
    # the interaction matrix
    A = 1/s*N@H*a

    #set all diagonal elements to -1 to ensure stability
    np.fill_diagonal(A,-1)
    # generate feasible growth rate
    r = np.random.uniform(0.00001,1, size = (num_species))
    ss = -np.linalg.inv(A)@r

    while not np.all(ss>=0):

        r = np.random.uniform(0.00001, 1., size=(num_species)) # changed max from 1 to 0.5 for stability of binary perts with few species
        ss = -np.linalg.inv(A) @ r


    C = np.random.uniform(-3,3, size = (num_species, num_pert)) * 1/s


    # for the binary pert scheme choose ICs to be close to the ss
    ICs = ss # this can be changed to start slightly away from ss
    return r, A, C, ICs


def binary_step_pert(t, pert_matrix, dt):
    i = min(int(t//dt), len(pert_matrix)-1) # solver sometimes goes slightly past end of time interval

    p = pert_matrix[i]
    return p

def generate_data_perts(simulator, tmax, sampling_time, dt, num_timecourses, ICs, num_pert, species_prob = 1, num_metabolites=0, noise_std = 0):
    ''''
    Generates data with external perturbations e.g. antibiotics or food.

    simulator: simulator object of the gMLV_sim class above
    tmax: max time (days)
    sampling_time: time between different perturbations
    dt: time between different simulated points
    num_timecourses:number of time courses to simulate
    ICs: intial conditions
    num_pert: number of different perturbations
    species_prob: probability of each species appearing in each timecourse
    num_metabolites: number of metabolites
    noise_std: standard dev of measruement noise
    '''

    ryobs = []  # species
    rsobs = []  # metabolites
    rysim = []
    rssim = []
    ry0 = []
    rs0 = []
    all_perts = []

    times = np.arange(0, tmax, dt)

    num_species = simulator.nsp


    for timecourse_idx in range(num_timecourses):
        if timecourse_idx%100 == 0:
            print('percent data generated:',timecourse_idx/num_timecourses * 100)

        pert_matrix = np.random.binomial(1, 0.5, size=(tmax//sampling_time, num_pert))


        all_perts.append(pert_matrix)

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(num_species,)) * ICs * np.random.binomial(1, species_prob, size=(num_species,))
        init_metabolites = np.random.uniform(low=10, high=50, size=num_metabolites)

        ysim, ssim, sy0, mu, M, _ = simulator.simulate(times=times, sy0=np.hstack((init_species, init_metabolites)),
                                                       p=lambda t: binary_step_pert(t, pert_matrix, sampling_time))
        if np.sum(ysim > 10)<0: # instability
            print('unstable')
        else:
            yobs = ysim[0:-1:int(sampling_time // dt)]
            sobs = ssim[0:-1:int(sampling_time // dt)]
            # add some gaussian noise
            yobs = yobs + np.random.normal(loc=0, scale=noise_std, size=yobs.shape)
            sobs = sobs + np.random.normal(loc=0, scale=noise_std, size=sobs.shape)

            # append results
            ryobs.append(yobs)
            rsobs.append(sobs)
            rysim.append(ysim)
            rssim.append(rssim)

            ry0.append(init_species)
            rs0.append(init_metabolites)
        # Xs, Fs = linearize_time_course_16S(yobs,times)
        # X = np.vstack([X, Xs])
        # F = np.vstack([F, Fs])

    ryobs = np.array(ryobs)
    rysim = np.array(rysim)
    all_perts = np.array(all_perts)

    return ryobs, rysim, all_perts


def generate_data_transplant(simulator, tmax, sampling_time, dt, num_timecourses, ICs, species_prob = 1, num_metabolites=0, noise_std = 0):
    ''''
        Generates data with transplant perturbations

        simulator: simulator object of the gMLV_sim class above
        tmax: max time (days)
        sampling_time: time between different perturbations
        dt: time between different simulated points
        num_timecourses:number of time courses to simulate
        ICs: intial conditions
        species_prob: probability of each species appearing in each timecourse
        num_metabolites: number of metabolites
        noise_std: standard dev of measruement noise
    '''


    ryobs = []  # species
    rsobs = []  # metabolites
    rysim = []
    rssim = []
    ry0 = []
    rs0 = []
    all_perts = []

    times = np.arange(0, sampling_time, dt)

    num_species = simulator.nsp


    for timecourse_idx in range(num_timecourses):

        if timecourse_idx%100 == 0:
            print('percent data generated:', timecourse_idx/num_timecourses * 100)

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(1, num_species)) * ICs * np.random.binomial(1, species_prob, size=(1, num_species))
        init_metabolites = np.random.uniform(low=10, high=50, size=(1,num_metabolites))

        ysim = []
        ssim = []

        p_matrix = []
        ys = init_species
        ss = init_metabolites
        yobs = [ys[0] + np.random.normal(loc=0, scale=noise_std, size=ys[0].shape)]
        sobs = [ss[0] + np.random.normal(loc=0, scale=noise_std, size=ss[0].shape)]

        p = np.zeros((num_species,))

        perturbed = False
        for i in range(int(tmax//sampling_time)):

            #print(yo.shape, ss.shape)


            ys, ss, sy0, mu, M, _ = simulator.simulate(times=times, sy0=np.hstack((ys[-1,:], ss[-1,:])))

            if np.random.uniform() < 0.1 and not perturbed and i < int(tmax//sampling_time)-1:
                perturbed = True

                p_rem = np.random.uniform(low=0, high=1, size=(num_species,)) * np.random.binomial(1,species_prob,
                                                                                                                  size=(
                                                                                                                  num_species,))

                p_add = np.random.uniform(low=0, high=1, size=(num_species,)) * np.random.binomial(1,species_prob,
                                                                                                                  size=(
                                                                                                                  num_species,))
                p = p_add - 2*p_rem
            else:
                p = np.zeros((num_species,))
            p_matrix.append(p)



            ys[-1, :] += p
            ys[ys < 0] = 0


            #print(yo.shape, ss.shape)
            yo = ys[-1]
            so = ss[-1]
            # add some gaussian noise

            yo = yo + np.random.normal(loc=0, scale=noise_std, size=yo.shape)
            so = so + np.random.normal(loc=0, scale=noise_std, size=so.shape)

            ysim.extend(ys)
            ssim.extend(ss)


            if i < int(tmax//sampling_time)-1:

                yobs.append(yo)
                sobs.append(so)


        all_perts.append(p_matrix)
        # append results
        ryobs.append(yobs)
        rsobs.append(sobs)
        rysim.append(ysim)
        rssim.append(rssim)

        ry0.append(init_species)
        rs0.append(init_metabolites)
        # Xs, Fs = linearize_time_course_16S(yobs,times)
        # X = np.vstack([X, Xs])
        # F = np.vstack([F, Fs])

    ryobs = np.array(ryobs)
    rysim = np.array(rysim)
    all_perts = np.array(all_perts)


    return ryobs, rysim, all_perts

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
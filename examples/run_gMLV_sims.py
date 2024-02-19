'''
Simulation script for gMLV model with perturbations and metabolites.
Usage: python run_gMLV_sims.py <save_path> <num_sims> <mode>
Example: python run_gMLV_sims.py outputs/ 100

The script will create a folder named outputs/ and save the results there.
The number of simulations is 100 by default, but can be changed by passing
a different number as the second argument.
'''

import logging
import random
import os
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')

import sys
sys.path.append("../")
sys.path.append("../gMLV")
from gMLV import *

# work around for retracing warning
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SMALL_SIZE = 13
MEDIUM_SIZE = 17
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def generate_params(num_species, num_pert, zero_prop=0, hetergeneous=False):
    '''
    generates parameters for GLV simulation according to Cao et al 2017
     (Inferring human microbial dynamics from temporal metagenomics data: Pitfalls and lessons)
     Method in the supplimentary
     num_species: number of microbial strains
     num_perterbations: number of perterbations
     zero_prop: proportion of the interaction matrix that should be zeros
    '''

    N = numpy.random.normal(0, 1, (num_species, num_species))

    if hetergeneous:
        y = 1.2
        u = numpy.random.uniform(0, 1, size=(num_species))
        H = (1-u)**(1/(1-y))
        H = numpy.diag(H)
        s = numpy.sum(H)
    else:
        H = numpy.eye(num_species)
        # s = 3 #from the paper
        s = numpy.sum(H)  # this seems to prevent instability when more species

    a = numpy.random.binomial(1, 1-zero_prop, size=(num_species, num_species))
    # the interaction matrix
    A = 1/s*N@H*a

    # set all diagonal elements to -1 to ensure stability
    numpy.fill_diagonal(A, -1)
    # generate feasible growth rate
    r = numpy.random.uniform(0.00001, 1, size=(num_species))
    ss = -numpy.linalg.inv(A)@r

    while not numpy.all(ss >= 0):

        # changed max from 1 to 0.5 for stability of binary perts with few species
        r = numpy.random.uniform(0.00001, 1., size=(num_species))
        ss = -numpy.linalg.inv(A) @ r

    C = numpy.random.uniform(-3, 3, size=(num_species, num_pert)) * 1/s

    # for the binary pert scheme choose ICs to be close to the ss
    ICs = ss  # this can be changed to start slightly away from ss
    return r, A, C, ICs

def generate_data_perts(simulator, tmax, sampling_time, dt, num_timecourses, ICs, num_pert, species_prob=1, num_metabolites=0, noise_std=0):
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

    times = numpy.arange(0, tmax, dt)

    num_species = simulator.nsp

    for timecourse_idx in range(num_timecourses):

        pert_matrix = numpy.random.binomial(1, 0.5, size=(tmax//sampling_time, num_pert))

        #print( "perturbations: ")
        #print(pert_matrix )

        all_perts.append(pert_matrix)

        # initial conditions
        init_species = numpy.random.uniform(low=0, high=2, size=(
            num_species,)) * ICs * numpy.random.binomial(1, species_prob, size=(num_species,))
        init_metabolites = numpy.random.uniform(
            low=10, high=50, size=num_metabolites)

        ysim, ssim, sy0, mu, M, _ = simulator.simulate(times=times, sy0=numpy.hstack((init_species, init_metabolites)),
                                                       u=lambda t: binary_step_pert(t, pert_matrix, sampling_time))
        if numpy.sum(ysim > 10) < 0:  # instability
            print('unstable')
        else:
            yobs = ysim[0:-1:int(sampling_time // dt)]
            sobs = ssim[0:-1:int(sampling_time // dt)]
            # add some gaussian noise
            yobs = yobs + \
                numpy.random.normal(loc=0, scale=noise_std, size=yobs.shape)
            sobs = sobs + \
                numpy.random.normal(loc=0, scale=noise_std, size=sobs.shape)

            # append results
            ryobs.append(yobs)
            rsobs.append(sobs)
            rysim.append(ysim)
            rssim.append(rssim)

            ry0.append(init_species)
            rs0.append(init_metabolites)
        # Xs, Fs = linearize_time_course_16S(yobs,times)
        # X = numpy.vstack([X, Xs])
        # F = numpy.vstack([F, Fs])

    ryobs = numpy.array(ryobs)
    rysim = numpy.array(rysim)
    all_perts = numpy.array(all_perts)

    return ryobs, rysim, all_perts


def generate_data_transplant(simulator, tmax, sampling_time, dt, num_timecourses, ICs, species_prob=1, num_metabolites=0, noise_std=0):
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

    times = numpy.arange(0, sampling_time, dt)

    num_species = simulator.nsp

    for timecourse_idx in range(num_timecourses):
        
        # initial conditions
        init_species = numpy.random.uniform(low=0, high=2, size=(
            1, num_species)) * ICs * numpy.random.binomial(1, species_prob, size=(1, num_species))
        init_metabolites = numpy.random.uniform(
            low=10, high=50, size=(1, num_metabolites))

        ysim = []
        ssim = []

        p_matrix = []
        ys = init_species
        ss = init_metabolites
        yobs = [
            ys[0] + numpy.random.normal(loc=0, scale=noise_std, size=ys[0].shape)]
        sobs = [
            ss[0] + numpy.random.normal(loc=0, scale=noise_std, size=ss[0].shape)]

        p = numpy.zeros((num_species,))

        perturbed = False
        for i in range(int(tmax//sampling_time)):

            # print(yo.shape, ss.shape)

            ys, ss, sy0, mu, M, _ = simulator.simulate(
                times=times, sy0=numpy.hstack((ys[-1, :], ss[-1, :])))

            if numpy.random.uniform() < 0.1 and not perturbed and i < int(tmax//sampling_time)-1:
                perturbed = True

                p_rem = numpy.random.uniform(low=0, high=1, size=(num_species,)) * numpy.random.binomial(1, species_prob,
                                                                                                   size=(
                                                                                                       num_species,))

                p_add = numpy.random.uniform(low=0, high=1, size=(num_species,)) * numpy.random.binomial(1, species_prob,
                                                                                                   size=(
                                                                                                       num_species,))
                p = p_add - 2*p_rem
            else:
                p = numpy.zeros((num_species,))
            p_matrix.append(p)

            ys[-1, :] += p
            ys[ys < 0] = 0

            # print(yo.shape, ss.shape)
            yo = ys[-1]
            so = ss[-1]
            # add some gaussian noise

            yo = yo + numpy.random.normal(loc=0, scale=noise_std, size=yo.shape)
            so = so + numpy.random.normal(loc=0, scale=noise_std, size=so.shape)

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
        # X = numpy.vstack([X, Xs])
        # F = numpy.vstack([F, Fs])

    ryobs = numpy.array(ryobs)
    rysim = numpy.array(rysim)
    all_perts = numpy.array(all_perts)

    return ryobs, rysim, all_perts

def binary_step_pert(t, pert_matrix, dt):
    # solver sometimes goes slightly past end of time interval
    i = min(int(t//dt), len(pert_matrix)-1)

    p = pert_matrix[i]
    return p


# some plotting functions
def plot_fit_gMLV_pert(yobs, yobs_h, perts, sobs, sobs_h, sampling_times, ysim, times):
    # plot the fit
    fig, axs = plt.subplots(1, 2, figsize=(16., 6.))

    for species_idx in range(yobs.shape[1]):
        axs[0].plot(times, ysim[:, species_idx], '--', label='simulation')

    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs[0].scatter(sampling_times, yobs[:, species_idx],
                       s=100, marker='x', label='observed')

    axs[0].set_prop_cycle(None)

    # for species_idx in range(yobs.shape[1]):
    #    axs[0].scatter(sampling_times, yobs_h[:, species_idx], s= 100,marker ='x', label = 'prediction')

    axs[0].set_xlabel('time (days)')
    axs[0].set_ylabel('[species]')

    handles, labels = axs[0].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)

    axs[0].legend(newHandles, newLabels)

    axs[1].set_prop_cycle(None)
    # perts = np.vstack((perts[0], perts[0], perts))
    # sampling_times = np.append(sampling_times, 100)

    for pert_idx in range(perts.shape[1]):
        axs[1].scatter(sampling_times[1:],
                       perts[:, pert_idx], marker='o', s=100)
        axs[1].set_xlim(left=0, right=100)

    axs[1].set_ylabel('transplant perturbation')
    axs[1].set_xlabel('time')

    # for metabolite_idx in range(sobs.shape[1]):
    #     axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
    #     axs[1].plot(timepoints, sobs_h[:, metabolite_idx], '--', color=cols[metabolite_idx])
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('[metabolite]');

# set_all_seeds(1234)


if __name__ == '__main__':

    if len(sys.argv) == 4:
        # check if the third argument is a number, and if the second one is a path
        if not sys.argv[2].isdigit():
            print("Please enter a valid number of simulations")
            sys.exit(1)
        if not os.path.isdir(sys.argv[1]):
            print("Please enter a valid path to save the outputs")
            sys.exit(1)

        num_sims = int(sys.argv[2])
        save_path = sys.argv[1] + '/'
        mode = int(sys.argv[3])
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Using default values for number of simulations and save path")
        print("Usage: python run_gMLV_sims.py <save_path> <num_sims> <mode: perturbation=0, transplant=1>")
        num_sims = 100
        save_path = 'outputs/'
        mode = 0
        os.makedirs(save_path, exist_ok=True)

    # set_all_seeds(0)

    # total number of time courses will be num_sims  x num_timecourses (per timecourse)
    num_timecourses = 1  # 9*100

    num_species = 3

    # controls probability of dropout
    species_prob = 1.0

    # npert is number of independent perturbations
    # FIXME: change num_pert to 0 and see fix issue with input array sizes or shapes
    num_pert = 1
    num_metabolites = 0

    # construct interaction matrix
    zero_prop = 0.  # the proportion of zeros in the interaction matrix

    tmax = 100
    sampling_time = 10
    dt = 1

    times = np.arange(0, tmax, dt)
    sampling_times = np.arange(0, tmax, sampling_time)

    # print("npert", num_pert)
    # print("nsims", num_timecourses)

    all_ryobs = np.zeros([num_sims, sampling_times.shape[0], num_species])
    all_rysim = np.zeros([num_sims, times.shape[0], num_species])
    all_parms = np.zeros([num_sims, num_species + num_species*num_species + num_species])

    if mode == 0:
        # This is parameter perturbations
        all_perts = np.zeros([num_sims, sampling_times.shape[0], num_pert])
    else:
        # This is transplant perturbations
        all_perts = np.zeros([num_sims, sampling_times.shape[0], num_species])

    for nsim in range(num_sims):
        # print("nsim",nsim)

        if nsim % 10 == 0:
            print('percent data generated:', nsim/num_sims * 100)

        # generate params according to paper approach
        #  C is perturbation interaction vector/m (also called epsilon)
        mu, M, C, ss = generate_params(num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)

        # print("mu: ", mu)
        # print("M: ", M)
        # print("C: ", C)

        all_parms[nsim, :] = np.concatenate(
            (mu.flatten(), M.flatten(), C.flatten()), axis=None)
        # print("p:", all_parms[nsim,:] )

        # instantiate simulator
        simulator = gMLV_sim(num_species=num_species,
                             num_metabolites=num_metabolites,
                             M=M,
                             mu=mu,
                             epsilon=C)

        if mode == 0:
            ryobs, rysim, perts = generate_data_perts(simulator, tmax, sampling_time, dt, num_timecourses, ss, num_pert, species_prob=species_prob, noise_std=0.00)

        else:
            ryobs, rysim, perts = generate_data_transplant(simulator, tmax, sampling_time, dt, num_timecourses, ss, species_prob=1, num_metabolites=0, noise_std=0.00)

        # print(ryobs.shape, rysim.shape, all_perts.shape)

        # species levels and perturbations for each time point
        all_ryobs[nsim, :, :] = ryobs.astype(np.float32)
        all_rysim[nsim, :, :] = rysim.astype(np.float32)
        # export each simulation as csv
        # create a numpy array concatenating the time points and the simulated data
        data_export = np.concatenate(
            (times.reshape(-1, 1), rysim[0, :, :]), axis=1)
        np.savetxt(save_path + '/simulations' + str(nsim) +
                   '.csv', data_export, delimiter=',')

        # np.savetxt(save_path + '/simulations' + str(nsim) + '.csv', rysim, delimiter=',')
        # np.savetxt(save_path + '/simulations.csv', rysim[0,:,:], delimiter=',')
        all_perts[nsim, :, :] = perts.astype(np.float32)

    np.save(save_path + '/abundances_sampled.npy', all_ryobs)
    np.save(save_path + '/abundances.npy', all_rysim)
    np.save(save_path + '/perts.npy', all_perts)
    np.save(save_path + '/parms.npy', all_parms)

    # plot some of the results
    for i in range(10):
        plot_fit_gMLV_pert(all_ryobs[i], 0,  # pred[-i-1, :, :],
                           all_perts[i, 0:-1, :], None, None, sampling_times, all_rysim[i], times)

        # print("new timecourse")
        # print( all_ryobs[i] )
        # print( all_perts[i, 0:-1, :] )

        plt.savefig(save_path + '/test_plot_' + str(i) + '.pdf')

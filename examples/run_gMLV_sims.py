'''
Simulation script for gMLV model with perturbations and metabolites.
Usage: python run_gMLV_sims.py <save_path> <num_sims>
Example: python run_gMLV_sims.py outputs/ 100

The script will create a folder named outputs/ and save the results there.
The number of simulations is 100 by default, but can be changed by passing
a different number as the second argument.
'''


import logging
from time import time
import math
from gMLV import *
from numpy import linalg as la
import copy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('tkagg')


sys.path.append('../')
# testing the the linter g;l

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

    if len(sys.argv) == 3:
        # check if the third argument is a number, and if the second one is a path
        if not sys.argv[2].isdigit():
            print("Please enter a valid number of simulations")
            sys.exit(1)
        if not os.path.isdir(sys.argv[1]):
            print("Please enter a valid path to save the outputs")
            sys.exit(1)

        num_sims = int(sys.argv[2])
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Using default values for number of simulations and save path")
        print("Usage: python run_gMLV_sims.py <save_path> <num_sims>")
        num_sims = 100
        save_path = 'outputs/'
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
    all_perts = np.zeros([num_sims, sampling_times.shape[0], num_pert])
    all_parms = np.zeros(
        [num_sims, num_species + num_species*num_species + num_species])

    for nsim in range(num_sims):
        # print("nsim",nsim)

        # QUESTION: what is the purpose of this if loop?
        if nsim % 100 == 0:
            print('percent data generated:', nsim/num_sims * 100)

        # generate params according to paper approach
        #  C is perturbation interaction vector/m
        # TODO: #25 generate_params is not defined anywhere
        mu, M, C, ss = generate_params(
            num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)

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
                             C=C)

        # FIXME: #26 generate_data_perts is not defined anywhere
        ryobs, rysim, perts = generate_data_perts(
            simulator, tmax, sampling_time, dt, num_timecourses, ss, num_pert, species_prob=species_prob, noise_std=0.00)

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

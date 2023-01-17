import random
import os
import sys
gMLV_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gMLV') # for some readson this is required to get lcluster to see gMLV
sys.path.append(gMLV_path)


import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import copy
from numpy import linalg as la
import os

sys.path.append('../')

from gMLV import *
from gMLV_sim import *
import math
from time import time

from RNN import get_RNN, custom_fit


# work around for retracing warning
import logging, os
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
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)



# some plotting functions
def plot_fit_gMLV_pert(yobs, yobs_h, perts, sobs, sobs_h, sampling_times, ysim, times):
    # plot the fit
    fig, axs = plt.subplots(1, 2, figsize = (16., 6.))



    for species_idx in range(yobs.shape[1]):
        axs[0].plot(times, ysim[:, species_idx], '--', label = 'simulation')


    axs[0].set_prop_cycle(None)

    #for species_idx in range(yobs.shape[1]):
    #   axs[0].scatter(sampling_times, yobs[:, species_idx], s=100, marker='x', label='observed')

    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs[0].scatter(sampling_times, yobs_h[:, species_idx], s= 100,marker ='x', label = 'prediction')


    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')

    handles, labels = axs[0].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)

    axs[0].legend(newHandles, newLabels)

    axs[1].set_prop_cycle(None)
    #perts = np.vstack((perts[0], perts[0], perts))
    #sampling_times = np.append(sampling_times, 100)

    for pert_idx in range(perts.shape[1]):
        axs[1].scatter(sampling_times[1:],  perts[:, pert_idx], marker='o', s=100)
        axs[1].set_xlim(left = 0, right = 100)


    axs[1].set_ylabel('transplant perturbation')
    axs[1].set_xlabel('time')


    # for metabolite_idx in range(sobs.shape[1]):
    #     axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
    #     axs[1].plot(timepoints, sobs_h[:, metabolite_idx], '--', color=cols[metabolite_idx])
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('[metabolite]');

#set_all_seeds(1234)



if __name__ == '__main__':
    set_all_seeds(0)

    num_species = 10
    species_prob = 1.
    num_pert = 3
    num_metabolites = 0

    # construct interaction matrix
    zero_prop = 0. # the proportion of zeros in the interaction matrix
    known_zero_prop = 0. # the proportion of the zeros that we know are zero

    mu, M, C, ss = generate_params(num_species, num_pert, zero_prop = zero_prop, hetergeneous=False)


    zeros = np.where(M==0)

    randomize = np.arange(M.shape[0])
    np.random.shuffle(randomize)

    if known_zero_prop > 0 and zero_prop > 0:
        known_zeros = [zeros[0][randomize][:int(len(zeros) * known_zero_prop)],
                   zeros[1][randomize][:int(len(zeros) * known_zero_prop)]]
    else:
        known_zeros = [[],[]]


    # instantiate simulator
    simulator = gMLV_sim(num_species=num_species,
                         num_metabolites=num_metabolites,
                         M=M,
                         mu=mu,
                         C=C)

    num_timecourses = 9*100
    tmax = 100
    n_epochs = 100

    val_prop = 0.1

    # best parameters from param scan
    L2_reg = 1e-7
    dy_dx_reg = 1e1
    GRU_size = 256

    if len(sys.argv) == 3: # usually for running on the cluster
        exp = int(sys.argv[2]) -1

        tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
        # inestigation scan over

        num_timecoursess = [96, 480, 96, 480]
        known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
        #species_probs = [0.1, 0.25, 0.5, 0.75, 1.]
        dy_dx_regs = [1e3, 1e2, 1e1, 1., 1e-1]

        num_timecourses = num_timecoursess[tc]
        known_zero_prop = known_zero_props[zp]
        #species_prob = species_probs[sp]
        species_prob = 1
        dy_dx_reg = dy_dx_regs[sp]

        save_path = sys.argv[1] + '/repeat' + sys.argv[2] + '/'
        n_epochs = 50 #number of epochs for largest dataset
        n_epochs = int(n_epochs*num_timecoursess[-1]/num_timecourses) # change n_epochs so that the same number of batch updates are run for each test

        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './working_dir'

    transplant_pert = True

    if transplant_pert:
        num_pert = num_species

    sampling_time = 10
    dt = 1

    if transplant_pert:
        ryobs, rysim, all_perts = generate_data_transplant(simulator, tmax, sampling_time, dt, num_timecourses, ss, species_prob=species_prob, noise_std=0.00)

    else:
        ryobs, rysim, all_perts = generate_data_perts(simulator, tmax, sampling_time, dt, num_timecourses, ss, num_pert, species_prob=species_prob, noise_std=0.00)


    print(ryobs.shape, rysim.shape, all_perts.shape)
    times = np.arange(0, tmax, dt)
    sampling_times = np.arange(0, tmax, sampling_time)


    ryobs = ryobs.astype(np.float32) # species levels and perturbations for each time point
    all_perts = all_perts.astype(np.float32)



    ## FIT RNN
    model = get_RNN(num_species, num_pert, len(sampling_times), GRU_size=GRU_size, L2_reg=L2_reg)
    train_preds, val_preds, train_loss, val_loss, model = custom_fit(model, ryobs, all_perts, known_zeros, n_epochs, val_prop, dy_dx_reg=dy_dx_reg, verbose=True)
    train_preds.extend(val_preds)

    pred = np.array(train_preds)

    np.save(save_path + '/data.npy', data)
    np.save(save_path + '/preds.npy', pred)
    np.save(save_path + '/test_loss.npy', val_loss)
    np.save(save_path + '/train_loss.npy', train_loss)
    model.save(save_path + '/RNN' ) # not working on cluster


    # plot some of the results
    for i in range(20):

        plot_fit_gMLV_pert(ryobs[-i-1], pred[-i-1, :, :],
                           all_perts[-i-1, 0:-1, :], None, None, sampling_times, rysim[-i-1], times)

        plt.savefig(save_path + '/test_plot_' + str(i) + '.pdf')

        plot_fit_gMLV_pert(ryobs[i], pred[i, :, :],
                           all_perts[i, 0:-1, :], None, None, sampling_times, rysim[i], times)
        plt.savefig(save_path + '/train_plot_' + str(i) + '.pdf')


    plt.figure()
    plt.plot(train_loss, label = 'train')
    plt.plot(val_loss, label = 'test')
    plt.ylim(ymin=0)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path + '/train_test_SSE.png', dpi=300)


    #plt.show()
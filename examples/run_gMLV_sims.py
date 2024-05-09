'''
Simulation script for gMLV model with perturbations and metabolites.
Usage: python run_sim_gMLVs.py <save_path> <num_sims> <mode>
Example: python run_sim_gMLVs.py outputs/ 100

The script will create a folder named outputs/ and save the results there.
The number of simulations is 100 by default, but can be changed by passing
a different number as the second argument.
'''

import logging
import os
import random
import sys
from typing import Any, List, Tuple, Optional, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import signedinteger
from numpy.typing import NDArray


from mimic.model_infer import *
from mimic.model_simulate import *

matplotlib.use('tkagg')

sys.path.append("../")

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


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def generate_params(num_species: int,
                    num_pert: int,
                    zero_prop: float = 0,
                    hetergeneous: bool = False) -> Tuple[NDArray[np.float64],
                                                         NDArray[np.float64],
                                                         NDArray[np.float64],
                                                         NDArray[np.float64]]:
    '''
    Generates parameters for GLV simulation according to Cao et al 2017.
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
        H = (1 - u) ** (1 / (1 - y))
        H = np.diag(H)
    else:
        H = np.eye(num_species)

    s = np.sum(H)
    a = np.random.binomial(1, 1 - zero_prop, size=(num_species, num_species))
    # the interaction matrix
    A = 1 / s * N @ H * a
    # set all diagonal elements to -1 to ensure stability
    np.fill_diagonal(A, -1)
    # generate feasible growth rate

    r = np.random.uniform(0.00001, 1, size=(num_species))
    ss = -np.linalg.inv(A) @ r

    while not np.all(ss >= 0):
        # changed max from 1 to 0.5 for stability of binary perts with few
        # species
        r = np.random.uniform(0.00001, 1., size=(num_species))
        ss = -np.linalg.inv(A) @ r

    C = np.random.uniform(-3, 3, size=(num_species, num_pert)) * 1 / s

    # for the binary pert scheme choose ICs to be close to the ss
    ICs = ss   # this can be changed to start slightly away from ss
    return r, A, C, ICs


def generate_data_perts(
    simulator: Any,
    tmax: int,
    sampling_time: int,
    dt: int,
    num_timecourses: int,
    ICs: NDArray[np.float64],
    num_pert: int,
    species_prob: float = 1,
    num_metabolites: int = 0,
    noise_std: float = 0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    ''''
    Generates data with external perturbations e.g. antibiotics or food.

    simulator: simulator object of the sim_gMLV class above
    tmax: max time (days)
    sampling_time: time between different perturbations
    dt: time between different simulated points
    num_timecourses: number of time courses to simulate
    ICs: intial conditions
    num_pert: number of different perturbations
    species_prob: probability of each species appearing in each timecourse
    num_metabolites: number of metabolites
    noise_std: standard dev of measruement noise
    '''

    ryobs: List[NDArray[np.float64]] = []  # species
    rsobs: List[Any] = []  # metabolites
    rysim: List[NDArray[np.float64]] = []
    rssim: List[Any] = []
    ry0: List[NDArray[np.float64]] = []
    rs0: List[Any] = []
    all_perts: List[Any] = []

    times = np.arange(0, tmax, dt)

    num_species = simulator.nsp

    for timecourse_idx in range(num_timecourses):

        pert_matrix = np.random.binomial(
            1, 0.5, size=(tmax // sampling_time, num_pert))

        all_perts.append(pert_matrix)

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(
            num_species,)) * ICs * np.random.binomial(1, species_prob, size=(num_species,))
        init_metabolites = np.random.uniform(
            low=10, high=50, size=num_metabolites)

        ysim, ssim, sy0, mu, M, _ = simulator.simulate(
            times=times, sy0=np.hstack(
                (init_species, init_metabolites)), u=lambda t: binary_step_pert(
                t, pert_matrix, sampling_time))
        if np.sum(ysim > 10) < 0:  # instability
            print('unstable')
        else:
            yobs = ysim[0:-1:int(sampling_time // dt)]
            sobs = ssim[0:-1:int(sampling_time // dt)]
            # add some gaussian noise
            yobs = yobs + \
                np.random.normal(loc=0, scale=noise_std, size=yobs.shape)
            sobs = sobs + \
                np.random.normal(loc=0, scale=noise_std, size=sobs.shape)

            # append results
            ryobs.append(yobs)
            rsobs.append(sobs)
            rysim.append(ysim)
            rssim.append(rssim)

            ry0.append(init_species)
            rs0.append(init_metabolites)

    ryobs_np = np.array(ryobs)
    ryobs_np = cast(NDArray[np.float64], ryobs_np)
    rysim_np = np.array(rysim)
    all_perts_np = np.array(all_perts)

    return ryobs_np, rysim_np, all_perts_np


def generate_data_transplant(
    simulator: Any,
    tmax: int,
    sampling_time: int,
    dt: int,
    num_timecourses: int,
    ICs: NDArray[np.float64],
    species_prob: float = 1.0,
    num_metabolites: int = 0,
    noise_std: float = 0.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Generates data with transplant perturbations.

    simulator: simulator object of the sim_gMLV class above
    tmax: max time (days)
    sampling_time: time between different perturbations
    dt: time between different simulated points
    num_timecourses: number of time courses to simulate
    ICs: intial conditions
    species_prob: probability of each species appearing in each timecourse
    num_metabolites: number of metabolites
    noise_std: standard dev of measruement noise
    '''

    ryobs: List[NDArray[np.float64]] = []  # species
    rsobs: List[Any] = []  # metabolites
    rysim: List[NDArray[np.float64]] = []
    rssim: List[Any] = []
    ry0: List[NDArray[np.float64]] = []
    rs0: List[Any] = []
    all_perts: List[Any] = []

    times = np.arange(0, sampling_time, dt)

    num_species = simulator.nsp

    for timecourse_idx in range(num_timecourses):

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(
            1, num_species)) * ICs * np.random.binomial(1, species_prob, size=(1, num_species))
        init_metabolites = np.random.uniform(
            low=10, high=50, size=(1, num_metabolites))

        ysim: List[Any] = []
        ssim: List[Any] = []

        p_matrix: List[Any] = []
        ys = init_species
        ss = init_metabolites
        yobs = [
            ys[0] + np.random.normal(loc=0, scale=noise_std, size=ys[0].shape)]
        sobs = [
            ss[0] + np.random.normal(loc=0, scale=noise_std, size=ss[0].shape)]

        p = np.zeros((num_species,))

        perturbed = False
        for i in range(int(tmax // sampling_time)):

            ys, ss, sy0, mu, M, _ = simulator.simulate(
                times=times, sy0=np.hstack((ys[-1, :], ss[-1, :])))

            if np.random.uniform() < 0.1 and not perturbed and i < int(tmax // sampling_time) - 1:
                perturbed = True

                p_rem = np.random.uniform(low=0, high=1, size=(
                    num_species,)) * np.random.binomial(1, species_prob, size=(num_species,))

                p_add = np.random.uniform(low=0, high=1, size=(
                    num_species,)) * np.random.binomial(1, species_prob, size=(num_species,))
                p = p_add - 2 * p_rem
            else:
                p = np.zeros((num_species,))
            p_matrix.append(p)

            ys[-1, :] += p
            ys[ys < 0] = 0

            yo = ys[-1]
            so = ss[-1]
            # add some gaussian noise

            yo = yo + np.random.normal(loc=0, scale=noise_std, size=yo.shape)
            so = so + np.random.normal(loc=0, scale=noise_std, size=so.shape)

            ysim.extend(ys)
            ssim.extend(ss)

            if i < int(tmax // sampling_time) - 1:
                yobs.append(yo)
                sobs.append(so)

        all_perts.append(p_matrix)
        # append results

        ryobs.append(np.array(yobs))
        rsobs.append(sobs)
        rysim.append(np.array(ysim))
        rssim.append(rssim)

        ry0.append(init_species)
        rs0.append(init_metabolites)

    ryobs_np = np.array(ryobs)
    rysim_np = np.array(rysim)
    all_perts_np = np.array(all_perts)

    return ryobs_np, rysim_np, all_perts_np


def binary_step_pert(t: float,
                     pert_matrix: NDArray[np.int_],
                     dt: int) -> NDArray[np.int_]:
    # solver sometimes goes slightly past end of time interval
    i = min(int(t // dt), len(pert_matrix) - 1)
    return pert_matrix[i]


# some plotting functions
def plot_fit_gMLV_pert(
    yobs: NDArray[np.float64],
    yobs_h: Optional[NDArray[np.float64]],
    perts: NDArray[np.float64],
    sobs: Optional[Any],
    sobs_h: Optional[Any],
    sampling_times: NDArray[np.int_],
    ysim: NDArray[np.float64],
    times: NDArray[signedinteger[Any]]
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16., 6.))

    for species_idx in range(yobs.shape[1]):
        axs[0].plot(times, ysim[:, species_idx], '--', label='simulation')

    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs[0].scatter(sampling_times, yobs[:, species_idx],
                       s=100, marker='x', label='observed')

    axs[0].set_prop_cycle(None)

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

    for pert_idx in range(perts.shape[1]):
        axs[1].scatter(sampling_times[1:],
                       perts[:, pert_idx], marker='o', s=100)
        axs[1].set_xlim(left=0, right=100)

    axs[1].set_ylabel('transplant perturbation')
    axs[1].set_xlabel('time')


if __name__ == '__main__':

    if len(sys.argv) == 4:
        # check if the third argument is a number, and if the second one is a
        # path
        if not sys.argv[2].isdigit():
            print("Please enter a valid number of simulations")
            sys.exit(1)
        if not os.path.isdir(sys.argv[1]):
            print("Please enter a valid path to save the outputs")
            sys.exit(1)

        num_sims = int(sys.argv[2])
        save_path = f'{sys.argv[1]}/'
        mode = int(sys.argv[3])
    else:
        print("Using default values for number of simulations and save path")
        print("Usage: python run_sim_gMLVs.py <save_path> <num_sims> <mode: perturbation=0, transplant=1>")
        num_sims = 100
        save_path = 'outputs/'
        mode = 0
    os.makedirs(save_path, exist_ok=True)
    # total number of time courses will be num_sims  x num_timecourses (per
    # timecourse)

    num_timecourses = 1

    num_species = 3

    # controls probability of dropout
    species_prob = 1.0
    # npert is number of independent perturbations
    num_pert = 1
    num_metabolites = 0

    # construct interaction matrix
    zero_prop = 0.  # the proportion of zeros in the interaction matrix

    tmax = 100
    sampling_time = 10
    dt = 1

    times = np.arange(0, tmax, dt)
    sampling_times = np.arange(0, tmax, sampling_time)

    all_ryobs = np.zeros(
        [num_sims, sampling_times.shape[0], num_species], dtype=np.float64)
    all_rysim = np.zeros(
        [num_sims, times.shape[0], num_species], dtype=np.float64)
    all_parms = np.zeros(
        [num_sims, num_species + num_species**2 + num_species], dtype=np.float64)

    if mode == 0:
        # This is parameter perturbations
        all_perts = np.zeros(
            [num_sims, sampling_times.shape[0], num_pert], dtype=np.float64)
    else:
        # This is transplant perturbations
        all_perts = np.zeros(
            [num_sims, sampling_times.shape[0], num_species], dtype=np.float64)

    for nsim in range(num_sims):

        if nsim % 10 == 0:
            print('percent data generated:', nsim / num_sims * 100)
        # generate params according to paper approach
        #  C is perturbation interaction vector/m (also called epsilon)
        mu, M, C, ss = generate_params(
            num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)

        all_parms[nsim, :] = np.concatenate(
            (mu.flatten(), M.flatten(), C.flatten()), axis=None)

        # instantiate simulator
        simulator = sim_gMLV(num_species=num_species,
                             num_metabolites=num_metabolites,
                             M=M,
                             mu=mu,
                             epsilon=C)

        if mode == 0:
            ryobs, rysim, perts = generate_data_perts(
                simulator, tmax, sampling_time, dt, num_timecourses, ss, num_pert, species_prob=species_prob, noise_std=0.00)
        else:
            ryobs, rysim, perts = generate_data_transplant(
                simulator, tmax, sampling_time, dt, num_timecourses, ss, species_prob=1, num_metabolites=0, noise_std=0.00)

        # species levels and perturbations for each time point
        all_ryobs[nsim, :, :] = ryobs.astype(np.float64)
        all_rysim[nsim, :, :] = rysim.astype(np.float64)
        # export each simulation as csv
        # create a np array concatenating the time points and the simulated
        # data
        data_export = np.concatenate(
            (times.reshape(-1, 1), rysim[0, :, :]), axis=1)
        np.savetxt(
            f'{save_path}/simulations{str(nsim)}.csv',
            data_export,
            delimiter=',',
        )

        all_perts[nsim, :, :] = perts.astype(np.float64)

    np.save(f'{save_path}/abundances_sampled.npy', all_ryobs)
    np.save(f'{save_path}/abundances.npy', all_rysim)
    np.save(f'{save_path}/perts.npy', all_perts)
    np.save(f'{save_path}/parms.npy', all_parms)

    # plot some of the results
    for i in range(10):
        plot_fit_gMLV_pert(all_ryobs[i], None, all_perts[i, 0:-1, :],
                           None, None, sampling_times, all_rysim[i], times)
        plt.savefig(f'{save_path}/test_plot_{str(i)}.pdf')

import json
from xdot import xdot
import time
import tensorflow as tf
from gMLV.gMLV_sim import *
from RED.environments.OED_env import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from casadi import *
import math
import sys
import os


sys.path.append('../')
sys.path.append('../../RED_master/')


mpl.use('tkagg')


def disablePrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def simulate(env, us, plot=False, calculate_FIM=True):
    # test simulation
    if calculate_FIM:
        print(DM(us))
        env.CI_solver = env.get_control_interval_solver(
            control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(
            N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, params, us)

        print('est_trajectory', est_trajectory)
        FIM = env.get_FIM(est_trajectory)
        FIM += DM(np.ones(FIM.size()) * eta)
        q, r = qr(FIM)
        logdet_FIM = trace(log(r))
        print('log det FIM:', logdet_FIM)

        est_trajectory = np.hstack(
            (np.array(env.initial_Y).reshape(-1, 1), est_trajectory))

    else:
        env.CI_solver = env.get_control_interval_solver(
            control_interval_time, dt, mode='sim')
        trajectory_solver = env.get_sampled_trajectory_solver(
            N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(y0, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        est_trajectory = np.hstack(
            (np.array(y0).reshape(3, 1), est_trajectory))

    if plot:
        t = np.arange(0, N_control_intervals+1)
        us = np.vstack((us[:, 0], us.T))
        fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.0))

        axs[0].step(t, us[:, 0], '--', alpha=1., linewidth=4, color='C0')
        axs[0].set_ylabel('Perturbation (AU)')
        axs[1].step(t, us[:, 1], '--', alpha=1., linewidth=4, color='C1')
        axs[1].set_ylabel('Perturbation (AU)')
        axs[2].step(t, us[:, 2], '--', alpha=1., linewidth=4, color='C2')
        axs[2].set_ylabel('Perturbation (AU)')

        plt.xlabel('Time (AU)')
        plt.ylabel('Perturbation (AU)')
        plt.savefig('./working_dir/perturbation.png', dpi=300)

        plt.figure()

        plt.plot(est_trajectory[0:n_observed_variables,
                 :].T, alpha=1, linewidth=4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Population (AU)')
        plt.savefig('./working_dir/population.png', dpi=300)

        plt.figure()
        plt.spy(FIM)
        param_labels = ['$M_{11}$', '$M_{12}$', '$M_{13}$', '$M_{21}$', '$M_{22}$', '$M_{23}$', '$M_{31}$', '$M_{32}$',
                        '$M_{33}$', '$\mu_1$', '$\mu_2$', '$\mu_3$', '$E_{11}$', '$E_{12}$', '$E_{13}$', '$E_{21}$',
                        '$E_{22}$', '$E_{23}$', '$E_{31}$', '$E_{32}$', '$E_{33}$']
        plt.xticks(range(0, 21), param_labels)
        plt.yticks(range(0, 21), param_labels)

        # plt.colorbar()

        plt.title('Estimated FIM')
        plt.savefig('./working_dir/FIM.png', dpi=300)

        plt.figure()

        cov = np.array(inv(FIM).elements()).reshape(FIM.size(), order='F')
        plt.spy(cov)
        # plt.colorbar()
        plt.title('Estimated covariance matrix')
        plt.xticks(range(0, 21), param_labels)
        plt.yticks(range(0, 21), param_labels)
        plt.savefig('./working_dir/Covariance matrix.png', dpi=300)

        param_variances = [math.sqrt(cov[i][i]) for i in range(cov.shape[0])]
        np.save('./working_dir/param_variances.npy', param_variances)
        print(params)
        print(param_variances)

        cov[cov < 0] = -1
        cov[cov > 0] = 1
        plt.figure()
        plt.imshow(cov)
        plt.colorbar()

        plt.show()

    return est_trajectory


if __name__ == '__main__':
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    set_all_seeds(0)

    gMLV_params = json.load(open('gMLV_params.json'))

    eta = gMLV_params['eta']

    params = np.load('working_dir/generated_params.npy')
    y0 = np.load('working_dir/generated_y0.npy')

    print('y0', y0)
    print('params:', params)

    theta = params
    n_species = 3
    M = theta[0:n_species ** 2].reshape(
        (n_species, n_species)).T  # transpose as casadi reshape is inverse compared to numpy

    gr = theta[n_species ** 2:n_species ** 2 + n_species]
    E = theta[n_species ** 2 + n_species:].reshape((n_species, n_species)).T

    print(M)
    print(gr)
    print(E)

    params = DM(params)

    print(params.size())
    actual_params = params
    N_control_intervals = 8
    control_interval_time = 1  # AU
    num_inputs = -1
    input_bounds = [[0, 1], [0, 1], [0, 1]]
    n_observed_variables = 3
    n_controlled_inputs = 3
    dt = 0.1
    normaliser = -1

    save_path = './'

    args = y0, xdot, params, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)

    # TODO: fix this hard-coded path
    us = np.load(
        '/Users/neythen/Desktop/Projects/gMLV/OED/results/OED_results_100822/rand_100_days/us.npy', allow_pickle=True)

    # us = np.array([[0,0,0], [1,0,0],[0,1,0],[0,0,1], [1,1,0],[1,0,1],[0,1,1],[1,1,1]]).T
    print(us)
    # us = np.random.rand(*us.shape)
    # np.save('/home/neythen/Desktop/Projects/gMLV/OED/results/OED_8_day_rational/random/us.npy', us)
    simulate(env, us, plot=True)

    calculate_FIM = True

    if calculate_FIM:
        env.CI_solver = env.get_control_interval_solver(
            control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(
            N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        print(est_trajectory)
        print(est_trajectory.shape)
        FIM = env.get_FIM(est_trajectory)
        FIM += DM(np.ones(FIM.size()) * eta)
        print(FIM)
        print(FIM.size())

        q, r = qr(FIM)

        det_FIM = np.prod(diag(r).elements())

        logdet_FIM = trace(log(r)).elements()[
            0]  # do it like this to protect from numerical errors from multiplying large EVs
        print('det FIM:', det_FIM)
        print('log det FIM:', logdet_FIM)
    else:
        env.CI_solver = env.get_control_interval_solver(
            control_interval_time, dt, mode='sim')
        trajectory_solver = env.get_sampled_trajectory_solver(
            N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(y0, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        est_trajectory = np.hstack(
            (np.array(y0).reshape(3, 1), est_trajectory))

        plt.plot(us.T, alpha=0.5, linewidth=4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Perturbation (AU)')
        plt.savefig('./working_dir/perturbation.png', dpi=300)
        plt.figure()

        plt.plot(est_trajectory.T, alpha=0.5, linewidth=4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Population (AU)')
        plt.savefig('./working_dir/population.png', dpi=300)

        plt.show()

import sys
import os



sys.path.append('../')
sys.path.append('../../RED_master/')

import math
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from RED.environments.OED_env import *
from gMLV.gMLV_sim import *

import tensorflow as tf
import time

from xdot import xdot
import json
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


def simulate(env, us, plot = False, calculate_FIM = True):
    # test simulation
    if calculate_FIM:
        print(DM(us))
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, params,us)

        print('est_trajectory', est_trajectory)
        FIM = env.get_FIM(est_trajectory)
        FIM += DM(np.ones(FIM.size()) * eta)
        q, r = qr(FIM)
        logdet_FIM = trace(log(r))
        print('log det FIM:', logdet_FIM)

        est_trajectory = np.hstack((np.array(env.initial_Y).reshape(-1, 1), est_trajectory))

    else:
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='sim')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(y0, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        est_trajectory = np.hstack((np.array(y0).reshape(3, 1), est_trajectory))

    if plot:
        t = np.arange(0, 101) * 2
        us = np.vstack((us[:, 0], us.T))
        plt.step(t, us[:, 0], '--', alpha=0.5, linewidth=4)
        plt.step(t, us[:, 1], '--', alpha=0.5, linewidth=4)
        plt.step(t, us[:, 2], '--', alpha=0.5, linewidth=4)

        plt.xlabel('Time (days)')
        plt.ylabel('Perturbation (AU)')
        plt.savefig('./working_dir/perturbation.png', dpi=300)
        plt.figure()

        plt.plot(est_trajectory[0:n_observed_variables,:].T, alpha=0.5, linewidth=4)
        plt.xlabel('Time (days)')
        plt.ylabel('Population (AU)')
        plt.savefig('./working_dir/population.png', dpi=300)

        plt.figure()
        plt.imshow(FIM)
        plt.colorbar()

        plt.title('FIM')
        plt.savefig('./working_dir/FIM.png', dpi=300)

        plt.figure()

        cov = np.array(inv(FIM).elements()).reshape(FIM.size(), order = 'F')
        plt.imshow(cov)
        plt.colorbar()
        plt.title('Covariance matrix')
        plt.savefig('./working_dir/Covariance matrix.png', dpi=300)

        param_variances = [math.sqrt(cov[i][i]) for i in range(cov.shape[0])]
        np.save('./working_dir/param_variances.npy', param_variances)
        print(params)
        print(param_variances)
        plt.show()

    return est_trajectory


if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    set_all_seeds(0)

    gMLV_params = json.load(open('gMLV_params.json'))

    eta = gMLV_params['eta']


    params = np.load('working_dir/generated_params.npy')
    y0 = np.load('working_dir/generated_y0.npy')

    print('y0', y0)
    print('params:', params)


    params = DM(params)

    print(params.size())
    actual_params = params
    N_control_intervals = 10
    control_interval_time = 1
    num_inputs = -1
    input_bounds = [[0,1], [0,1], [0,1]]
    n_observed_variables = 3
    n_controlled_inputs = 3
    dt = 0.1
    normaliser = -1


    save_path = './'


    args = y0, xdot, params, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)



    us = np.load('working_dir/us.npy')

    #us = np.random.rand(*us.shape)

    simulate(env, us, plot=True)











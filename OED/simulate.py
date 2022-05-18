import sys
import os




sys.path.append('../../RED_master/')

import math
from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from RED.environments.OED_env import *

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


def simulate(us, plot = False, calculate_FIM = True):
    # test simulation




    if calculate_FIM:
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        est_trajectory = np.hstack((np.array(env.initial_Y).reshape(-1, 1), est_trajectory))

        print(est_trajectory)
        print(est_trajectory.shape)
        FIM = env.get_FIM(est_trajectory)
        FIM += DM(np.ones(FIM.size()) * eta)
        print(FIM)
        print('trace FIM:', trace(FIM))
        print('eigen values: ', np.linalg.eig(FIM)[0], np.sum(np.linalg.eig(FIM)[0]))
        print(FIM.size())

        q, r = qr(FIM)

        det_FIM = np.prod(diag(r).elements())

        logdet_FIM = trace(log(r)).elements()[
            0]  # do it like this to protect from numerical errors from multiplying large EVs
        print('det FIM:', det_FIM)
        print('log det FIM:', logdet_FIM)

    else:
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='sim')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(y0, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))

        est_trajectory = np.hstack((np.array(y0).reshape(3, 1), est_trajectory))

    if plot:
        plt.plot(us.T, alpha=0.5, linewidth=4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Perturbation (AU)')
        plt.savefig('./working_dir/perturbation.png', dpi=300)
        plt.figure()

        plt.plot(est_trajectory[0:n_observed_variables,:].T, alpha=0.5, linewidth=4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Population (AU)')
        plt.savefig('./working_dir/population.png', dpi=300)

        plt.figure()
        plt.spy(FIM)
        plt.xticks(range(21),range(0,21))


        plt.show()


if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    y0 = np.array([1.,1.,1.])

    gMLV_params = json.load(open('gMLV_params.json'))
    M = np.array(gMLV_params['M'])
    E = np.array(gMLV_params['E'])
    gr = np.array(gMLV_params['gr'])
    y0 = np.array(gMLV_params['y0'])
    scale_factor = gMLV_params['scale_factor']
    eta = gMLV_params['eta']



    M *= scale_factor
    E *= scale_factor
    gr*= scale_factor
    y0 *= scale_factor

    params = np.hstack((M.flatten(), gr.flatten(), E.flatten())) # need to flatten for FIM calc
    print(params)




    params = DM(params)

    print(params.size())
    actual_params = params
    N_control_intervals = 10
    control_interval_time = 10 # AU
    num_inputs = -1
    input_bounds = [[0,1], [0,1], [0,1]]
    n_observed_variables = 3
    n_controlled_inputs = 3
    dt = 0.01
    normaliser = -1


    save_path = './'


    args = y0, xdot, params, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)

    us = np.random.rand(3, N_control_intervals)
    #us = np.zeros((3,N_control_intervals))*scale_factor
    #us = np.array([[0,0,1]]*N_control_intervals).T*scale_factor

    us = np.load('working_dir/us.npy').T
    simulate(us, plot=True)











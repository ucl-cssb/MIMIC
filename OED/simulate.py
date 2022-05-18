import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')
sys.path.append('/Users/neythen/Desktop/Projects/RED_master/')

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



if __name__ == '__main__':
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    eta = 0.01 # small constant to add to each FIM element ot prevent singularity


    scale_factor = 0.1
    M = np.ones((3,3)) * -scale_factor
    gr = np.ones((3,1)) * scale_factor
    E = np.eye(3) * -scale_factor # 3 antibiotics, one for each bacteria


    # M = np.array([
    #     [-0.1, 0, 0],
    #     [0, -0.1, 0],
    #     [0, 0, -0.1]
    # ])

    gr = np.array([
        [1],
        [11],
        [12]
    ])*scale_factor

    y0 = [0.1,0.1,0.1]

    params = np.hstack((M.flatten(), gr.flatten(), E.flatten())) # need to flatten for FIM calc
    print(params)


    params = DM(params)

    print(params.size())
    actual_params = params
    N_control_intervals = 10
    control_interval_time = 1 # AU
    num_inputs = -1
    input_bounds = [[0,1], [0,1], [0,1]]
    n_observed_variables = 3
    n_controlled_inputs = 3
    dt = 1
    normaliser = -1


    save_path = './'


    args = y0, xdot, params, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time,normaliser
    env = OED_env(*args)




    # test simulation
    us = np.random.rand(3, N_control_intervals)*scale_factor
    #us = np.zeros((3,N_control_intervals))*scale_factor
    #us = np.array([[0,0,1]]*N_control_intervals).T*scale_factor

    calculate_FIM = True

    if calculate_FIM:
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
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
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='sim')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(y0, params,
                                           reshape(us, (n_controlled_inputs, N_control_intervals)))


        est_trajectory = np.hstack((np.array(y0).reshape(3,1), est_trajectory))


        plt.plot(us.T, alpha = 0.5, linewidth = 4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Perturbation (AU)')
        plt.savefig('./working_dir/perturbation.png', dpi = 300)
        plt.figure()

        plt.plot(est_trajectory.T, alpha = 0.5, linewidth = 4)
        plt.xlabel('Time (AU)')
        plt.ylabel('Population (AU)')
        plt.savefig('./working_dir/population.png', dpi=300)

        plt.show()












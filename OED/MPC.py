import sys
import os

IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'imports')

sys.path.append(IMPORT_PATH)
sys.path.append('/Users/neythen/Desktop/Projects/ROCC/')
sys.path.append('/Users/neythen/Desktop/Projects/RED_master/')

import math
from casadi import *
import numpy as np
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

    M = np.ones((3, 3)) * -0.1
    gr = np.ones((3, 1))
    E = np.eye(3) * -1  # 3 antibiotics, one for each bacteria

    # M = np.array([
    #     [-0.1, 0, 0],
    #     [0, -0.1, 0],
    #     [0, 0, -0.1]
    # ])

    gr = np.array([
        [1],
        [1.1],
        [1.2]
    ])

    y0 = [1, 1, 1]

    params = np.hstack((M.flatten(), gr.flatten(), E.flatten()))  # need to flatten for FIM calc
    print(params)

    params = DM(params)


    actual_params = params
    N_control_intervals = 10
    control_interval_time = 12 # AU

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
    us = np.random.rand(3, 10)
    us = np.zeros((3, 10))
    # us = np.array([[0,0,1]]*10).T

    trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
    est_trajectory = trajectory_solver(env.initial_Y, param_guesses,
                                       reshape(us, (n_controlled_inputs, N_control_intervals)))


    print(est_trajectory)



    env.u0 = DM(u0)



    def get_full_u_solver():
        us = SX.sym('us', N_control_intervals * n_controlled_inputs)
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, param_guesses, reshape(us , (n_controlled_inputs, N_control_intervals)))

        print(est_trajectory.shape)
        FIM = env.get_FIM(est_trajectory)
        print(FIM.shape)
        q, r = qr(FIM)

        obj = -trace(log(r))
        # obj = -log(det(FIM))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = True) # for some reason limited mem works better for the MPC
        # solver.print_options()
        # sys.exit()

        return solver


    u0 = [(input_bounds[1] - input_bounds[0]) / 2] * n_controlled_inputs*N_control_intervals
    u_solver = get_full_u_solver()
    sol = u_solver(x0=u0, lbx = [input_bounds[0]]*n_controlled_inputs*N_control_intervals, ubx = [input_bounds[1]]*n_controlled_inputs*N_control_intervals)
    us = sol['x']
    print(sol)




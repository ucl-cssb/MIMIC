import sys
import os

sys.path.append('../../RED_master/')
sys.path.append('../')
import math
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from RED.environments.OED_env import *
from gMLV import *
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



if __name__ == '__main__':
    set_all_seeds(0)

    gMLV_params = json.load(open('gMLV_params.json'))

    eta = gMLV_params['eta']
    num_pert = 3
    num_species = 3
    zero_prop = 0


    gr, M, E, y0 = generate_params(num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)



    print(gr.shape, M.shape, E.shape, y0.shape)
    print(E)

    params = np.hstack((M.flatten(), gr.flatten(), E.flatten()))  # need to flatten for FIM calc

    np.save('working_dir/generated_params.npy', params)
    np.save('working_dir/generated_y0.npy', y0)

    print(params.shape)

    params = DM(params)

    print(params.size())
    actual_params = params
    N_control_intervals = 100
    control_interval_time = 10 # in days
    num_inputs = -1
    input_bounds = [[0, 1], [0, 1], [0, 1]]
    n_observed_variables = 3
    n_controlled_inputs = 3
    dt = 0.1
    normaliser = -1

    save_path = './'

    args = y0, xdot, params, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)




    def get_full_u_solver():
        us = SX.sym('us', N_control_intervals * n_controlled_inputs)
        env.CI_solver = env.get_control_interval_solver(control_interval_time, dt, mode='OED')
        trajectory_solver = env.get_sampled_trajectory_solver(N_control_intervals, control_interval_time, dt)
        est_trajectory = trajectory_solver(env.initial_Y, actual_params, reshape(us , (n_controlled_inputs, N_control_intervals)))

        print('est_trajectory', est_trajectory.shape)
        FIM = env.get_FIM(est_trajectory)
        FIM += DM(np.ones(FIM.size()) * eta)
        print(FIM.shape)
        q, r = qr(FIM)

        obj = -trace(log(r))
        # obj = -log(det(FIM))
        nlp = {'x': us, 'f': obj}
        solver = env.gauss_newton(obj, nlp, us, limited_mem = False) # for some reason limited mem works better for the MPC
        # solver.print_options()
        # sys.exit()

        return solver


    u0 = [0.5] * n_controlled_inputs*N_control_intervals
    env.u0 = DM(u0)
    u_solver = get_full_u_solver()
    sol = u_solver(x0=u0, lbx = [0]*n_controlled_inputs*N_control_intervals, ubx = [1]*n_controlled_inputs*N_control_intervals)
    us = sol['x']
    print(sol)
    print(reshape(us , (n_controlled_inputs, N_control_intervals)))


    us = np.array(us.elements()).reshape(n_controlled_inputs,  N_control_intervals, order = 'F')
    print('logdetFIM:', -sol['f'])
    print('us:')
    print(us)


    np.save('working_dir/us.npy', us)







import sys
import os

IMPORT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(IMPORT_PATH)
print(IMPORT_PATH)
IMPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'RED_master')

sys.path.append(IMPORT_PATH)

print(IMPORT_PATH)
from RED.environments.OED_env import OED_env
from RED.agents.continuous_agents import RT3D_agent


from casadi import *
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

import time
import tensorflow as tf

import multiprocessing
import json
import math
from gMLV import *
from gMLV.gMLV_sim import *
from xdot import xdot



if __name__ == '__main__':
    set_all_seeds(0)
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    n_cores = multiprocessing.cpu_count()
    print('Num CPU cores:', n_cores)


    params = json.load(open('RL_params.json'))

    n_episodes, skip, input_bounds, n_controlled_inputs, num_inputs, dt, N_control_intervals, control_interval_time, n_observed_variables, prior, normaliser = \
        [params[k] for k in params.keys()]


    num_pert = 3
    num_species = 3
    zero_prop = 0

    gr, M, E, y0 = generate_params(num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)



    sim_params = np.hstack((M.flatten(), gr.flatten(), E.flatten()))  # need to flatten for FIM calc

    actual_params = DM(sim_params)

    normaliser = np.array(normaliser)

    n_params = actual_params.size()[0]
    n_system_variables = len(y0)
    n_FIM_elements = sum(range(n_params + 1))
    n_tot = n_system_variables + n_params * n_system_variables + n_FIM_elements

    param_guesses = actual_params

    if len(sys.argv) == 3:

        if int(sys.argv[2]) <= 10:
            prior = False
        else:
            prior = True
        # for parameter scan
        '''
        exp = int(sys.argv[2]) - 1
        # 3 learning rates
        # 4 hl sizes
        # 3 repeats per combination
        n_repeats = 3
        comb = exp // n_repeats
        pol_learning_rate = pol_learning_rates[comb//len(hidden_layer_sizes)]
        hidden_layer_size = hidden_layer_sizes[comb%len(hidden_layer_sizes)]
        '''

        save_path = sys.argv[1] + sys.argv[2] + '/'
        print(n_episodes)
        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './working_results'

    test_episode = False
    recurrent = True

    # these chosen from parameter scan
    pol_learning_rate = 0.00005
    hidden_layer_size = [[64, 64], [128, 128]]

    if recurrent:
        # pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs, [32, 32], [64,64,64], n_controlled_inputs]
        pol_layer_sizes = [n_observed_variables + 1, n_observed_variables + 1 + n_controlled_inputs,
                           hidden_layer_size[0], hidden_layer_size[1], n_controlled_inputs]
        val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs,
                           n_observed_variables + 1 + n_controlled_inputs, hidden_layer_size[0], hidden_layer_size[1],
                           1]
        # agent = DQN_agent(layer_sizes=[n_observed_variables + n_params + n_FIM_elements + 2, 100, 100, num_inputs ** n_controlled_inputs])
    else:
        pol_layer_sizes = [n_observed_variables + 1, 0, [], [128, 128], n_controlled_inputs]
        val_layer_sizes = [n_observed_variables + 1 + n_controlled_inputs, 0, [], [128, 128], 1]

    # agent = DRPG_agent(layer_sizes=layer_sizes, learning_rate = 0.0004, critic = True)
    agent = RT3D_agent(val_layer_sizes=val_layer_sizes, pol_layer_sizes=pol_layer_sizes, policy_act=tf.nn.sigmoid,
                       val_learning_rate=0.0001, pol_learning_rate=pol_learning_rate)  # , pol_learning_rate=0.0001)
    agent.batch_size = int(N_control_intervals * skip)
    agent.max_length = N_control_intervals + 1
    agent.mem_size = 500000000

    args = y0, xdot, param_guesses, actual_params, n_observed_variables, n_controlled_inputs, num_inputs, input_bounds, dt, control_interval_time, normaliser
    env = OED_env(*args)

    unstable = 0

    max_std = 1  # for exploring
    explore_rate = max_std
    alpha = 1
    # n_episodes = 10000
    env.mapped_trajectory_solver = env.CI_solver.map(skip, "thread", n_cores)
    total_t = time.time()

    n_unstables = []
    all_returns = []
    all_test_returns = []
    agent.std = 0.1
    agent.noise_bounds = [-0.25, 0.25]
    agent.action_bounds = [0, 1]
    policy_delay = 2
    update_count = 0
    fitted = False
    print('time:', control_interval_time)
    for episode in range(int(n_episodes // skip)):
        print(episode)

        if prior:
            # TODO:: generate these skip times
            actual_params = []
            for i in range(skip):
                gr, M, E, y0 = generate_params(num_species, num_pert, zero_prop=zero_prop, hetergeneous=False)
                sim_params = np.hstack((M.flatten(), gr.flatten(), E.flatten()))  # need to flatten for FIM calc
                actual_params.append(sim_params)

        else:

            actual_params = np.repeat([sim_params], skip, axis = 0)


        env.param_guesses = DM(actual_params)

        states = [env.get_initial_RL_state_parallel() for i in range(skip)]

        e_returns = [0 for _ in range(skip)]

        e_actions = []

        e_exploit_flags = []
        e_rewards = [[] for _ in range(skip)]
        e_us = [[] for _ in range(skip)]
        trajectories = [[] for _ in range(skip)]

        sequences = [[[0] * pol_layer_sizes[1]] for _ in range(skip)]

        env.reset()
        env.param_guesses = DM(actual_params)
        env.logdetFIMs = [[] for _ in range(skip)]
        env.detFIMs = [[] for _ in range(skip)]

        for e in range(0, N_control_intervals):

            if recurrent:
                inputs = [states, sequences]
            else:
                inputs = [states]

            if episode < 100 // skip:
                actions = agent.get_actions(inputs, explore_rate=1, test_episode=test_episode)
            else:
                actions = agent.get_actions(inputs, explore_rate=explore_rate, test_episode=test_episode)

            e_actions.append(actions)

            outputs = env.map_parallel_step(np.array(actions).T, actual_params, continuous=True)
            next_states = []

            for i, o in enumerate(outputs):
                next_state, reward, done, _, u = o
                e_us[i].append(u)
                next_states.append(next_state)
                state = states[i]

                action = actions[i]

                if e == N_control_intervals - 1 or np.all(np.abs(next_state) >= 1) or math.isnan(np.sum(next_state)):
                    # next_state = [0]*pol_layer_sizes[0] # maybe dont need this
                    done = True

                transition = (state, action, reward, next_state, done)
                trajectories[i].append(transition)
                sequences[i].append(np.concatenate((state, action)))
                if reward != -1:  # dont include the unstable trajectories as they override the true return
                    e_rewards[i].append(reward)
                    e_returns[i] += reward


            # print('sequences', np.array(sequences).shape)
            # print('sequences', sequences[0])
            states = next_states

        if test_episode:
            trajectories = trajectories[:-1]

        for trajectory in trajectories:
            if np.all([np.all(np.abs(trajectory[i][0]) <= 1) for i in range(len(trajectory))]) and not math.isnan(
                    np.sum(trajectory[-1][0])):  # check for instability
                agent.memory.append(trajectory)

            else:
                unstable += 1
                print('UNSTABLE!!!')
                print((trajectory[-1][0]))


        if episode > 100 // skip:
            print('training', update_count)
            t = time.time()
            for hello in range(skip):
                # print(e, episode, hello, update_count)
                update_count += 1
                policy = update_count % policy_delay == 0

                agent.Q_update(policy=policy, fitted=fitted, recurrent=recurrent, low_mem = True)
            print('fitting time', time.time() - t)

        explore_rate = agent.get_rate(episode, 0, 1, n_episodes / (11 * skip)) * max_std
        '''
        if episode > 1000//skip:
            update_count += 1
            agent.Q_update( policy=update_count%policy_delay == 0, fitted=True)
        '''

        print('n unstable ', unstable)
        n_unstables.append(unstable)

        if test_episode:
            all_returns.extend(e_returns[:-1])
            all_test_returns.append(np.sum(np.array(e_rewards)[-1, :]))
        else:
            all_returns.extend(e_returns)

        print()
        print('EPISODE: ', episode, episode * skip)

        print('moving av return:', np.mean(all_returns[-10 * skip:]))
        print('explore rate: ', explore_rate)
        print('alpha:', alpha)
        print('av return: ', np.mean(all_returns))
        print()

        # print('us:', np.array(e_us)[0, :])

        print('actions:', np.array(e_actions).shape)
        print('actions:', np.array(e_actions)[:, 0])

        print('return:', np.sum(np.array(e_rewards)[0, :]))
        print()

        if test_episode:
            print('test actions:', np.array(e_actions)[:, -1])
            print('test rewards:', np.array(e_rewards)[-1, :])
            print('test return:', np.sum(np.array(e_rewards)[-1, :]))
            print()

    print('time:', time.time() - total_t)
    print(env.detFIMs[-1])
    print(env.logdetFIMs[-1])
    np.save(save_path + '/all_returns.npy', np.array(all_returns))
    if test_episode:
        np.save(save_path + '/all_test_returns.npy', np.array(all_test_returns))

    np.save(save_path + '/n_unstables.npy', np.array(n_unstables))
    np.save(save_path + '/actions.npy', np.array(agent.actions))
    agent.save_network(save_path)

    # np.save(save_path + 'values.npy', np.array(agent.values))
    t = np.arange(N_control_intervals) * int(control_interval_time)

    plt.plot(all_test_returns)
    plt.figure()
    plt.plot(all_returns)
    plt.show()


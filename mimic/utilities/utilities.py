import json
import os

import random
import numpy as np
import matplotlib.pyplot as plt

cols = ["red", "green", "blue", "royalblue", "orange", "black"]


def plot_gLV(yobs, timepoints):
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 1)
    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
    axs.set_xlabel('time')
    axs.set_ylabel('[species]')


def plot_gMLV(yobs, sobs, timepoints):
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 2)
    for species_idx in range(yobs.shape[1]):
        axs[0].plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')
    if sobs.shape[1] > 0:
        for metabolite_idx in range(sobs.shape[1]):
            axs[1].plot(timepoints, sobs[:, metabolite_idx],
                        color=cols[metabolite_idx])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('[metabolite]')


def plot_fit_gMLV(yobs, yobs_h, sobs, sobs_h, timepoints):
    # plot the fit
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 2)

    for species_idx in range(yobs.shape[1]):
        axs[0].plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
        axs[0].plot(timepoints, yobs_h[:, species_idx],
                    '--', color=cols[species_idx])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')

    for metabolite_idx in range(sobs.shape[1]):
        axs[1].plot(timepoints, sobs[:, metabolite_idx],
                    color=cols[metabolite_idx])
        axs[1].plot(timepoints, sobs_h[:, metabolite_idx],
                    '--', color=cols[metabolite_idx])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('[metabolite]')


def plot_fit_gLV(yobs, yobs_h, sobs, sobs_h, timepoints):
    # plot the fit
    # fig, axs = plt.subplots(1, 2, layout='constrained')
    fig, axs = plt.subplots(1, 1)

    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
        axs.plot(timepoints, yobs_h[:, species_idx],
                 '--', color=cols[species_idx])
    axs.set_xlabel('time')
    axs.set_ylabel('[species]')


def compare_params(mu=None, M=None, alpha=None, e=None):
    # each argument is a tuple of true and predicted values (mu, mu_hat)
    if mu is not None:
        print("mu_hat/mu:")
        print(np.array(mu[1]))
        print(np.array(mu[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
                np.array(mu[1]), markerfmt="D", label='mu_hat', linefmt='C0-')
        ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
                np.array(mu[0]), markerfmt="X", label='mu', linefmt='C1-')
        ax.set_xlabel('i')
        ax.set_ylabel('mu[i]')
        ax.legend()

    if M is not None:
        print("\nM_hat/M:")
        print(np.round(np.array(M[1]), decimals=2))
        print("\n", np.array(M[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, M[0].shape[0] ** 2),
                np.array(M[1]).flatten(), markerfmt="D", label='M_hat', linefmt='C0-')
        ax.stem(np.arange(0, M[0].shape[0] ** 2),
                np.array(M[0]).flatten(), markerfmt="X", label='M', linefmt='C1-')
        ax.set_ylabel('M[i,j]')
        ax.legend()

    if alpha is not None:
        print("\na_hat/a:")
        print(np.round(np.array(alpha[1]), decimals=2))
        print("\n", np.array(alpha[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, alpha[0].shape[0] * alpha[0].shape[1]),
                np.array(alpha[1]).flatten(), markerfmt="D", label='a_hat', linefmt='C0-')
        ax.stem(np.arange(0, alpha[0].shape[0] * alpha[0].shape[1]),
                np.array(alpha[0]).flatten(), markerfmt="X", label='a', linefmt='C1-')
        ax.set_ylabel('a[i,j]')
        ax.legend()

    if e is not None:
        print("\ne_hat/e:")
        print(np.round(np.array(e[1]), decimals=2))
        print("\n", np.array(e[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, e[0].shape[0]), np.array(
            e[1]).flatten(), markerfmt="D", label='e_hat', linefmt='C0-')
        ax.stem(np.arange(0, e[0].shape[0]), np.array(
            e[0]).flatten(), markerfmt="X", label='e', linefmt='C1-')
        ax.set_ylabel('e[i]')
        ax.legend()


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def read_parameters(json_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, json_file)
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters

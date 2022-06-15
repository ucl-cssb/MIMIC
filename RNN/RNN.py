import random
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from scipy.integrate import odeint
import copy
from numpy import linalg as la
import os
import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gMLV import *

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)



# some plotting functions



def plot_gMLV(yobs, sobs, timepoints):
    fig, axs = plt.subplots(1, 2)
    for species_idx in range(yobs.shape[1]):
        axs[0].plot(timepoints, yobs[:, species_idx], color=cols[species_idx])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')
    if sobs.shape[1] > 0:
        for metabolite_idx in range(sobs.shape[1]):
            axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('[metabolite]');

def plot_fit_gMLV(yobs, yobs_h, sobs, sobs_h, timepoints):
    # plot the fit
    fig, axs = plt.subplots(1, 1)

    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx], label = 'simulation')

    plt.gca().set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs_h[:, species_idx], '--', label = 'prediction')

    axs.set_xlabel('time')
    axs.set_ylabel('[species]')

    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)

    # for metabolite_idx in range(sobs.shape[1]):
    #     axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
    #     axs[1].plot(timepoints, sobs_h[:, metabolite_idx], '--', color=cols[metabolite_idx])
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('[metabolite]');

def compare_params(mu=None, M=None, alpha=None, e=None):
    # each argument is a tuple of true and predicted values
    if mu is not None:
        print("mu_hat/mu:")
        print(np.array(mu[1]))
        print(np.array(mu[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0,len(mu[0]), dtype="int32"), np.array(mu[1]), markerfmt="D")
        ax.stem(np.arange(0,len(mu[0]), dtype="int32"), np.array(mu[0]), markerfmt="X")
        ax.set_xlabel('i')
        ax.set_ylabel('mu[i]');

    if M is not None:
        print("\nM_hat/M:")
        print(np.round(np.array(M[1]), decimals=2))
        print("\n",np.array(M[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, M[0].shape[0] ** 2), np.array(M[1]).flatten(), markerfmt="D")
        ax.stem(np.arange(0, M[0].shape[0] ** 2), np.array(M[0]).flatten(), markerfmt="X")
        ax.set_ylabel('M[i,j]');

    if alpha is not None:
        print("\na_hat/a:")
        print(np.round(np.array(alpha[1]), decimals=2))
        print("\n",np.array(alpha[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, alpha[0].shape[0] * alpha[0].shape[1]), np.array(alpha[1]).flatten(), markerfmt="D")
        ax.stem(np.arange(0, alpha[0].shape[0] * alpha[0].shape[1]), np.array(alpha[0]).flatten(), markerfmt="X")
        ax.set_ylabel('a[i,j]');

    if e is not None:
        print("\ne_hat/e:")
        print(np.round(np.array(e[1]), decimals=2))
        print("\n",np.array(e[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, e[0].shape[0]), np.array(e[1]).flatten(), markerfmt="D")
        ax.stem(np.arange(0, e[0].shape[0]), np.array(e[0]).flatten(), markerfmt="X")
        ax.set_ylabel('e[i]');


def generate_params(num_species, num_pert, hetergeneous = False):

    '''
    generates parameters for GLV simulation according to Cao et al 2017
    '''

    N = np.random.normal(0, 1, (num_species, num_species))

    if hetergeneous:
        y = 1.2
        u = np.random.uniform(0, 1, size=(num_species))
        H = (1-u)**(1/(1-y))
        H = np.diag(H)
        s = np.sum(H)
    else:
        H = np.eye(num_species)
        s = 1

    a = np.random.binomial(1, 0.8, size=(num_species, num_species))


    # the interaction matrix
    A = 1/s*N@H*a

    #set all diagonal elements to -1 to ensure stability
    np.fill_diagonal(A,-1)

    # generate feasible growth rate
    r = np.random.uniform(0.001,1, size = (num_species))
    ss = -np.linalg.inv(A)@r

    while not np.all(ss>0):
        r = np.random.uniform(0.001, 1, size=(num_species))
        ss = -np.linalg.inv(A) @ r


    C = np.random.uniform(-3,3, size = (num_species, num_pert))


    # for the binary pert scheme choose ICs to be close to the ss
    ICs = ss # this can be change to start slightly away from ss

    return A, C, ICs

#A, C, ICs = generate_params(3,2,hetergeneous=False)



#set_all_seeds(1234)

## SETUP MODEL
# establish size of model
num_species = 10
num_metabolites = 0

# construct interaction matrix
#TODO do this programmatically
M = np.zeros((num_species, num_species))
np.fill_diagonal(M, [-0.05, -0.1, -0.15, -0.01, -0.2])
M[0, 2] = -0.025
M[1, 3] = 0.05
M[4, 0] = 0.02

# construct growth rates matrix
mu = np.random.lognormal(0.01, 0.5, num_species)

# instantiate simulator
simulator = gMLV_sim(num_species=num_species,
                     num_metabolites=num_metabolites,
                     M=M,
                     mu=mu)
simulator.print()

## PRODUCE SIMULATED RESULTS
# data structures for results
ryobs = [] # species
rsobs = [] # metabolites
ry0 = []
rs0 = []
X = np.array([], dtype=np.double).reshape(0, num_species+1)
F = np.array([], dtype=np.double).reshape(0, num_species)

num_timecourses = 320
tmax = 10
dt = 1
times = np.arange(0,10,dt)
for timecourse_idx in range(num_timecourses):
    # initial conditions
    init_species = np.random.uniform(low=10, high=50, size=num_species)
    init_metabolites = np.random.uniform(low=10, high=50, size=num_metabolites)

    yobs, sobs, sy0, mu, M, _ = simulator.simulate(times=times, sy0=np.hstack((init_species, init_metabolites)))

    # add some gaussian noise
    yobs = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)
    sobs = sobs + np.random.normal(loc=0, scale=0.1, size=sobs.shape)

    # append results
    ryobs.append(yobs)
    rsobs.append(sobs)
    ry0.append(init_species)
    rs0.append(init_metabolites)
    Xs, Fs = linearize_time_course_16S(yobs,times)
    X = np.vstack([X, Xs])
    F = np.vstack([F, Fs])

ryobs = np.array(ryobs)
print(ryobs.shape)
print(f"X: {X.shape}")
print(f"F: {F.shape}")
print(f"n: {num_species*F.shape[0]}, p: {num_species + num_species**2}")

inputs = copy.deepcopy(ryobs[:,:-1,:])

#inputs[:, 1:, :] = 0 #rmove everything apart from ICs in inputs

targets = copy.deepcopy(ryobs[:,1:,:])


## FIT RNN
model = keras.Sequential()


print(inputs.shape, targets.shape) # (n_simeseries, n_timepoints, n_species)

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(keras.Input(shape = (len(times)-1, num_species), name = "S_input"))
model.add(layers.GRU(32, return_sequences=True))

model.add(layers.Dense(num_species))

model.compile(optimizer = keras.optimizers.Adam(), loss = 'mse')
model.build()
print(model.summary())
history = model.fit(inputs, targets, verbose = True, batch_size = 32, epochs = 5000, validation_split=0.1)

#print(history.history)

pred = model.predict(inputs)
print(pred.shape)

for i in range(10):
    plot_fit_gMLV(np.vstack((inputs[-i,0,:][np.newaxis,:],targets[-i,:,:])), np.vstack((inputs[-i,0,:][np.newaxis,:],pred[-i,:,:])),None, None, times)
    plt.savefig('working_dir/plot_'+str(i) + '.png', dpi = 300)

plt.figure()
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()

plt.show()
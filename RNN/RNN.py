import random
import os
import sys
gMLV_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gMLV') # for some readson this is required to get lcluster to see gMLV
sys.path.append(gMLV_path)


import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import copy
from numpy import linalg as la
import os

sys.path.append('../')
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gMLV import *
from gMLV_sim import *
import math
from time import time




# work around for retracing warning
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)



# some plotting functions



def plot_gMLV(yobs, sobs, timepoints):
    fig, axs = plt.subplots(1,1)
    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx])
    axs.set_xlabel('time')
    axs.set_ylabel('[species]')
    # if sobs.shape[1] > 0:
    #     for metabolite_idx in range(sobs.shape[1]):
    #         axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
    #     axs[1].set_xlabel('time')
    #     axs[1].set_ylabel('[metabolite]');

def plot_fit_gMLV(yobs, yobs_h, sobs, sobs_h, timepoints):
    # plot the fit
    fig, axs = plt.subplots(1, 1)

    for species_idx in range(yobs.shape[1]):
        axs.plot(timepoints, yobs[:, species_idx])

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


def plot_fit_gMLV_pert(yobs_h, perts, sobs, sobs_h, sampling_times, ysim, times):
    # plot the fit
    fig, axs = plt.subplots(1, 2, figsize = (16., 6.))



    for species_idx in range(ysim.shape[1]):
        axs[0].plot(sampling_times, ysim[:, species_idx], '--', label = 'simulation')


    axs[0].set_prop_cycle(None)

    #for species_idx in range(yobs.shape[1]):
    #    axs[0].scatter(sampling_times, yobs[:, species_idx],s= 100,marker ='o', label = 'simulation')

    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs_h.shape[1]):
        axs[0].scatter(sampling_times, yobs_h[:, species_idx], s= 100,marker ='x', label = 'prediction')

    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')

    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)


    perts = np.vstack((perts[0], perts))

    for pert_idx in range(perts.shape[1]):
        axs[1].step(sampling_times, perts[:, pert_idx], '--')


    axs[1].set_ylabel('perturbation')
    axs[1].set_xlabel('time')


    # for metabolite_idx in range(sobs.shape[1]):
    #     axs[1].plot(timepoints, sobs[:, metabolite_idx], color=cols[metabolite_idx])
    #     axs[1].plot(timepoints, sobs_h[:, metabolite_idx], '--', color=cols[metabolite_idx])
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('[metabolite]');


def plot_fit_gMLV_pert(yobs, yobs_h, perts, sobs, sobs_h, sampling_times, ysim, times):
    # plot the fit
    fig, axs = plt.subplots(1, 2, figsize = (16., 6.))



    for species_idx in range(yobs.shape[1]):
        axs[0].plot(times, ysim[:, species_idx], '--', label = 'simulation')


    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs[0].scatter(sampling_times, yobs[:, species_idx], s=100, marker='x', label='observed')

    axs[0].set_prop_cycle(None)

    for species_idx in range(yobs.shape[1]):
        axs[0].scatter(sampling_times, yobs_h[:, species_idx], s= 100,marker ='o', label = 'prediction')


    axs[0].set_xlabel('time')
    axs[0].set_ylabel('[species]')

    handles, labels = axs[0].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)

    axs[0].legend(newHandles, newLabels)


    perts = np.vstack((perts[0], perts))

    for pert_idx in range(perts.shape[1]):
        axs[1].step(sampling_times, perts[:, pert_idx], '--')


    axs[1].set_ylabel('perturbation')
    axs[1].set_xlabel('time')


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



#set_all_seeds(1234)


def get_RNN(num_species, num_pert, num_ts, GRU_size=32, L2_reg = 0.):
    model = keras.Sequential()

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(keras.Input(shape=(num_ts - 1, num_species + num_pert), name="S_input", batch_size = batch_size))

    #model.add(layers.Dense(100, use_bias = False)) # 'embedding' layer

    model.add(layers.GRU(GRU_size, return_sequences=True, unroll = True, stateful = True))

    model.add(layers.GRU(GRU_size, return_sequences=True, unroll = True, stateful = True, kernel_regularizer=regularizers.L2(L2_reg)))

    model.add(layers.Dense(num_species, kernel_regularizer=regularizers.L2(L2_reg)))

    #model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    return model

## SETUP MODEL
# establish size of model

@tf.function
def run_batch(model, opt, batch_data, train = True, dy_dx_reg = 1e-5):

    #t_inputs = batch_inputs
    #batch_inputs = tf.Variable(batch_inputs, dtype=float)  # (32, 1, 10)
    # targets = tf.Variable(targets[start:end], dtype = float)



    with tf.GradientTape(persistent = True) as model_tape:

        # get trainable variables
        train_vars = model.trainable_variables
        # Create empty gradient list (not a tf.Variable list)
        accum_grad = [tf.zeros_like(this_var) for this_var in train_vars]

        abundances = batch_data[:, 0:1, :num_species] #get ICs
        all_preds = [abundances]
        total_pred_loss = 0.
        total_reg_loss = 0.

        for i in range(int(tmax//sampling_time)-1):

            perts = batch_data[:, i:i+1, num_species:]  # get perts for the first timestep

            batch_inputs = tf.concat([abundances, perts], 2)

            with tf.GradientTape(persistent=True) as loss_tape:
                loss_tape.watch(batch_inputs)

                #tf.print('timestep', i)
                pred = model(batch_inputs)  # (32, 1, 10)
            all_preds.append(pred)
            # first get gradients of pred wrt inputs
            dy_dx = loss_tape.batch_jacobian(pred,
                                             batch_inputs)  #  (batch, time, species, time, species+perts)

            dy_dx_loss = 0.
            for n in range(len(known_zeros[0])):
                #tf.print(known_zeros[1][n], 0, known_zeros[0][n])
                #tf.print(dy_dx.shape)
                #tf.print(dy_dx[:, 0, known_zeros[1][n], 0, known_zeros[0][n]].shape)
                dy_dx_loss = tf.add(dy_dx_loss, tf.reduce_mean(tf.square(
                    dy_dx[:, 0, known_zeros[1][n], 0, known_zeros[0][n]])))  # [all_batches, 1, n_sp, 1, n_sp]


            targets = batch_data[:, i+1:i+2, :num_species] # get the targets for this time point
            #loss = tf.add(tf.math.reduce_mean(tf.square(pred - targets)), tf.multiply(dy_dx_reg, dy_dx_loss))
            total_pred_loss = tf.add(total_pred_loss, tf.math.reduce_mean(tf.square(pred - targets)))
            total_reg_loss = tf.add(total_reg_loss, tf.multiply(dy_dx_reg, dy_dx_loss))
            #loss_grad = model_tape.gradient(loss, model.trainable_variables)
            #accum_grad = [(a_grad + grad/int(tmax//sampling_time)) for a_grad, grad in zip(accum_grad, loss_grad)] # accumulate the mean gradient
            abundances = pred

        total_loss = tf.divide(tf.add(total_reg_loss, total_pred_loss), batch_size)
        #

        # print('loss time', time() - t)

    if train:

        #tf.print( loss_grad)
        # print('grad time', time() - t)

        grad = model_tape.gradient(total_loss, train_vars)

        opt.apply_gradients(zip(grad, train_vars))

    model.reset_states()

    return all_preds,  total_loss


def run_epoch(model, opt, data, train = True, dy_dx_reg = 1e-5):
    batch_losses = []
    n_batches = data.shape[0] // batch_size
    all_preds = []
    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size


        preds, batch_loss = run_batch(model, opt, data[start:end],  dy_dx_reg=dy_dx_reg, train = train)

        preds = np.squeeze(np.array(preds), 2)

        preds = np.swapaxes(preds, 0, 1)
        all_preds.extend(preds)


        batch_losses.append(batch_loss)
        # print('opt time', time() - t)

    return all_preds, batch_losses

def custom_fit(model, data, val_prop, dy_dx_reg = 1e-5, verbose=False):


    # round split to multiple of batch size, assuming there is a multiple of batch size simulations then both training and valudation will be also
    split = int((1 - val_prop) * len(inputs))
    split = batch_size * math.floor(split/batch_size)

    train_data = data[:split]
    val_data = data[split:]

    opt = keras.optimizers.Adam()

    training_losses = []
    validation_losses = []

    for epoch in range(n_epochs):

        train_preds, train_losses = run_epoch(model, opt, train_data, train=True, dy_dx_reg=dy_dx_reg)
        val_preds, val_losses = run_epoch(model, opt, val_data, train=False, dy_dx_reg=dy_dx_reg)

        training_losses.append(tf.math.reduce_mean(train_losses))

        validation_losses.append(tf.math.reduce_mean(val_losses))

        if verbose:
            tf.print('epoch:', epoch, training_losses[-1], validation_losses[-1])

    return train_preds, val_preds, training_losses, validation_losses, model

if __name__ == '__main__':
    set_all_seeds(0)

    num_species = 10
    species_prob =0.5
    num_pert = 0
    num_metabolites = 0

    # construct interaction matrix
    zero_prop = 0.
    known_zero_prop = 0

    mu, M, C, ss = generate_params(num_species, num_pert, zero_prop = zero_prop, hetergeneous=False)


    zeros = np.where(M==0)



    randomize = np.arange(M.shape[0])
    np.random.shuffle(randomize)

    if known_zero_prop > 0 and zero_prop > 0:
        known_zeros = [zeros[0][randomize][:int(len(zeros) * known_zero_prop)],
                   zeros[1][randomize][:int(len(zeros) * known_zero_prop)]]
    else:
        known_zeros = [[],[]]

    '''
    # TODO:: remove these lines
    eye = np.eye(M.shape[0])
    zeros = np.where(eye == 0)
    known_zeros = [zeros[0],
                   zeros[1]]
    print(known_zeros)
    '''

    # construct growth rates matrix


    # instantiate simulator
    simulator = gMLV_sim(num_species=num_species,
                         num_metabolites=num_metabolites,
                         M=M,
                         mu=mu,
                         C=C)
    #simulator.print()

    num_timecourses = 96*3


    tmax = 100
    n_epochs = 400
    batch_size = 32

    val_prop = 0.1

    # training set and testing set have to be a multiple of batch size

    # custom training loop to incorporate prior knowledge
    L2_regs = [1e-8, 1e-7, 1e-6, 1e-5]
    GRU_sizes = [32, 64, 128, 256, 512]
    dy_dx_regs = [1e-2, 1e-3, 1e-4]

    # best parameters from param scan
    L2_reg = 1e-7
    dy_dx_reg = 1e1
    #dy_dx_reg = 0.
    GRU_size = 256

    if len(sys.argv) == 3:
        exp = int(sys.argv[2]) -1



        tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
        # inestigation scan over

        num_timecoursess = [96, 480, 96, 480]
        known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
        #species_probs = [0.1, 0.25, 0.5, 0.75, 1.]
        dy_dx_regs = [1e3, 1e2, 1e1, 1., 1e-1]

        num_timecourses = num_timecoursess[tc]
        known_zero_prop = known_zero_props[zp]
        #species_prob = species_probs[sp]
        species_prob = 1
        dy_dx_reg = dy_dx_regs[sp]

        save_path = sys.argv[1] + '/repeat' + sys.argv[2] + '/'
        n_epochs = 50 #number of epochs for largest dataset
        n_epochs = int(n_epochs*num_timecoursess[-1]/num_timecourses) # change n_epochs so that the same number of batch updates are run for each test

        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './working_dir'

    print(num_timecourses, known_zero_prop, species_prob, n_epochs)

    transplant_pert = True

    if transplant_pert:
        num_pert = num_species
    sampling_time = 10
    dt = 1

    if transplant_pert:
        ryobs, rysim, all_perts = generate_data_transplant(simulator, tmax, sampling_time, dt, num_timecourses, ss, species_prob=species_prob, noise_std=0.05)

        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/ryobs.npy', ryobs)
        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/rysim.npy', rysim)
        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/perts.npy', all_perts)

        # ryobs = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/ryobs.npy')[:num_timecourses]
        # rysim = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/rysim.npy')[:num_timecourses]
        # all_perts = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/transplant_pert/perts.npy')[:num_timecourses]
    else:
        ryobs, rysim, all_perts = generate_data_perts()

        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/ryobs.npy', ryobs)
        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/rysim.npy', rysim)
        #np.save('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/perts.npy', all_perts)

        # ryobs = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/ryobs.npy')[:num_timecourses]
        # rysim = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/rysim.npy')[:num_timecourses]
        # all_perts = np.load('/home/neythen/Desktop/Projects/gMLV/OED/training_data/antibiotic_pert/perts.npy')[:num_timecourses]

    times = np.arange(0, tmax, dt)
    sampling_times = np.arange(0, tmax, sampling_time)

    inputs = copy.deepcopy(ryobs[:,:-1,:])
    print(inputs.shape, all_perts.shape)
    #inputs[:, 1:, :] = 0 #rmove everything apart from ICs in inputs

    #inputs = inputs[:,0:1,:] # change to inputting each timestep sequentially for dy_dx calculation
    #print(all_perts.shape)
    #inputs = np.concatenate((inputs, all_perts[:, 0:1, :]), axis = 2).astype(np.float32)

    #print(inputs.shape)
    targets = copy.deepcopy(ryobs[:,1:,:]).astype(np.float32)

    # add 0 perturbation to the end so that shapes match
    all_perts = np.concatenate((all_perts, np.zeros(all_perts[:, 0:1, :].shape)), axis = 1)

    data = np.concatenate((ryobs, all_perts), axis = 2).astype(np.float32) # species levels and perturbations for each time point


    print(data.shape) # (n_simeseries, n_timepoints, n_species)

    ## FIT RNN
    print(len(sampling_times))




    print(L2_reg, GRU_size, dy_dx_reg)




    model = get_RNN(num_species, num_pert, len(sampling_times), GRU_size=GRU_size, L2_reg=L2_reg)
    train_preds, val_preds, train_loss, val_loss, model = custom_fit(model, data, val_prop, dy_dx_reg=dy_dx_reg, verbose=True)
    train_preds.extend(val_preds)

    pred = np.array(train_preds)



    np.save(save_path + '/data.npy', data)
    np.save(save_path + '/preds.npy', pred)
    np.save(save_path + '/val_loss.npy', val_loss)
    np.save(save_path + '/train_loss.npy', train_loss)
    #model.save(save_path + '/RNN' ) # not working on cluster

    #history = model.fit(inputs, targets, verbose = True, batch_size = batch_size, epochs = n_epochs, validation_split=0.1)

    #print(history.history)


    print(pred.shape)
    print(rysim.shape)
    for i in range(10):
        # print(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])))
        # print(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],pred[-i,:,:])))
        # plot_fit_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])), np.vstack((inputs[-i,0,:num_species][np.newaxis,:],pred[-i,:,:])),None, None, times)
        plot_fit_gMLV_pert(ryobs[-i-1], pred[-i-1, :, :],
                           all_perts[-i-1, 0:-1, :], None, None, sampling_times, rysim[-i-1], times)
        # plot_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])),None, times)
        plt.savefig(save_path + '/test_plot_' + str(i) + '.png', dpi=300)

        plot_fit_gMLV_pert(ryobs[i], pred[i, :, :],
                           all_perts[i, 0:-1, :], None, None, sampling_times, rysim[i], times)
        # plot_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])),None, times)
        plt.savefig(save_path + '/train_plot_' + str(i) + '.png', dpi=300)


    plt.figure()
    plt.plot(train_loss, label = 'train')
    plt.plot(val_loss, label = 'test')
    plt.ylim(ymin=0)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path + '/train_test_SSE.png', dpi=300)


    #plt.show()
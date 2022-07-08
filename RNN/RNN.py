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
        #s = 1 from the paper
        s = np.sum(H) # to prevent instability when more species

    a = np.random.binomial(1, 1-zero_prop, size=(num_species, num_species))


    # the interaction matrix
    A = 1/s*N@H*a



    #set all diagonal elements to -1 to ensure stability
    np.fill_diagonal(A,-1)


    # generate feasible growth rate
    r = np.random.uniform(0.00001,1, size = (num_species))
    ss = -np.linalg.inv(A)@r

    while not np.all(ss>=0):
        r = np.random.uniform(0.00001, 1., size=(num_species)) # changed max from 1 to 0.5 for stability of binary perts with few species
        ss = -np.linalg.inv(A) @ r


    C = np.random.uniform(-3,3, size = (num_species, num_pert)) * 1/s


    # for the binary pert scheme choose ICs to be close to the ss
    ICs = ss # this can be change to start slightly away from ss

    return r, A, C, ICs


def get_RNN(num_species, num_pert, num_ts, GRU_size=32, L2_reg = 0.):
    model = keras.Sequential()

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(keras.Input(shape=(num_ts - 1, num_species + num_pert), name="S_input"))

    #model.add(layers.Dense(100, use_bias = False)) # 'embedding' layer

    #model.add(layers.GRU(256, return_sequences=True))

    model.add(layers.GRU(GRU_size, return_sequences=True, unroll = True, kernel_regularizer=regularizers.L2(L2_reg)))

    model.add(layers.Dense(num_species, kernel_regularizer=regularizers.L2(L2_reg)))

    #model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    return model


def binary_step_pert(t, pert_matrix, dt):
    i = min(int(t//dt), len(pert_matrix)-1) # solver sometimes goes slightly past end of time interval

    p = pert_matrix[i]
    return p

def generate_data_perts():
    ryobs = []  # species
    rsobs = []  # metabolites
    rysim = []
    rssim = []
    ry0 = []
    rs0 = []
    all_perts = []

    times = np.arange(0, tmax, dt)


    for timecourse_idx in range(num_timecourses):
        if timecourse_idx%100 == 0:
            print(timecourse_idx/num_timecourses * 100)

        # generate binary perturbations matrix
        pert_matrix = np.random.binomial(1, 0.5, size=(tmax//sampling_time-1, num_pert))
        #pert_matrix = np.zeros((tmax // sampling_time - 1, num_pert))

        all_perts.append(pert_matrix)

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(num_species,)) * ICs * np.random.binomial(1, species_prob, size=(num_species,))
        init_metabolites = np.random.uniform(low=10, high=50, size=num_metabolites)

        ysim, ssim, sy0, mu, M, _ = simulator.simulate(times=times, sy0=np.hstack((init_species, init_metabolites)),
                                                       p=lambda t: binary_step_pert(t, pert_matrix, sampling_time))
        if np.sum(ysim > 10)<0: # instability
            print('unstable')
        else:
            yobs = ysim[0:-1:int(sampling_time // dt)]
            sobs = ssim[0:-1:int(sampling_time // dt)]
            # add some gaussian noise
            yobs = yobs + np.random.normal(loc=0, scale=noise_std, size=yobs.shape)
            sobs = sobs + np.random.normal(loc=0, scale=noise_std, size=sobs.shape)

            # append results
            ryobs.append(yobs)
            rsobs.append(sobs)
            rysim.append(ysim)
            rssim.append(rssim)

            ry0.append(init_species)
            rs0.append(init_metabolites)
        # Xs, Fs = linearize_time_course_16S(yobs,times)
        # X = np.vstack([X, Xs])
        # F = np.vstack([F, Fs])

    ryobs = np.array(ryobs)
    rysim = np.array(rysim)
    all_perts = np.array(all_perts)

    return ryobs, rysim, all_perts


def generate_data_transplant():
    ryobs = []  # species
    rsobs = []  # metabolites
    rysim = []
    rssim = []
    ry0 = []
    rs0 = []
    all_perts = []

    times = np.arange(0, sampling_time, dt)


    for timecourse_idx in range(num_timecourses):
        # generate binary perturbations matrix
        # pert_matrix = np.random.binomial(1, 0.5, size=(tmax//sampling_time-1, num_pert
        #                                               ))
        if timecourse_idx%100 == 0:
            print(timecourse_idx/num_timecourses * 100)

        # initial conditions
        init_species = np.random.uniform(low=0, high=2, size=(1, num_species)) * ICs * np.random.binomial(1, species_prob, size=(1, num_species))
        init_metabolites = np.random.uniform(low=10, high=50, size=(1,num_metabolites))

        ysim = []
        ssim = []

        p_matrix = []
        ys = init_species
        ss = init_metabolites
        yobs = [ys[0]]
        sobs = [ss[0]]

        p = np.zeros((num_species,))
        perturbed = False
        for i in range(int(tmax//sampling_time)):

            #print(yo.shape, ss.shape)

            ys, ss, sy0, mu, M, _ = simulator.simulate(times=times, sy0=np.hstack((ys[-1,:], ss[-1,:])))

            ys[-1, :] += p
            ys[ys < 0] = 0


            #print(yo.shape, ss.shape)
            yo = ys[-1]
            so = ss[-1]
            # add some gaussian noise

            yo = yo + np.random.normal(loc=0, scale=noise_std, size=yo.shape)
            so = so + np.random.normal(loc=0, scale=noise_std, size=so.shape)

            ysim.extend(ys)
            ssim.extend(ss)


            if i < int(tmax//sampling_time)-1:

                yobs.append(yo)
                sobs.append(so)

                if np.random.uniform() < 0.1 and not perturbed:
                    perturbed = True

                    #p_rem = np.random.uniform(low=-1, high=0, size=(num_species,))
                    p_add = np.random.uniform(low=0, high=1, size=(num_species,)) * ICs * np.random.binomial(1, species_prob, size=(num_species, ))
                    p =  p_add
                else:
                    p = np.zeros((num_species,))
                p_matrix.append(p)

        all_perts.append(p_matrix)
        # append results
        ryobs.append(yobs)
        rsobs.append(sobs)
        rysim.append(ysim)
        rssim.append(rssim)

        ry0.append(init_species)
        rs0.append(init_metabolites)
        # Xs, Fs = linearize_time_course_16S(yobs,times)
        # X = np.vstack([X, Xs])
        # F = np.vstack([F, Fs])

    ryobs = np.array(ryobs)
    rysim = np.array(rysim)
    all_perts = np.array(all_perts)

    return ryobs, rysim, all_perts
#set_all_seeds(1234)

## SETUP MODEL
# establish size of model

@tf.function
def run_batch(model, opt, batch_inputs, batch_targets, train = True, dy_dx_reg = 1e-5):

    #t_inputs = batch_inputs
    #batch_inputs = tf.Variable(batch_inputs, dtype=float)  # (32, 9, 10)
    # targets = tf.Variable(targets[start:end], dtype = float)
    with tf.GradientTape() as model_tape:
        with tf.GradientTape() as loss_tape:
            loss_tape.watch(batch_inputs)
            pred = model(batch_inputs)  # (32, 9, 10)
            # print(t_inputs.shape)
            # print(pred.shape)

            # print(loss_tape.gradient(pred[0], t_inputs[1]))
        # dy_dx = loss_tape.gradient(pred, t_inputs, unconnected_gradients=tf.UnconnectedGradients.ZERO) # (32, 9, 10)
        dy_dx = loss_tape.batch_jacobian(pred, batch_inputs)#[0, :, [0,1,2] , :, [1, 2, 3]]  # (32, 9, 10, 9, 10)

        #dy_dx = tf.gather_nd(dy_dx[:, :, [1, 2, 3, 0, 0, 4], :, [0, 0, 4, 1, 2, 3] ])


        dy_dx = tf.gather(dy_dx, known_zeros[1], axis = 2)


        dy_dx = tf.gather(dy_dx,known_zeros[0], axis= 4)



        #print(tf.math.reduce_mean(tf.square(dy_dx)), tf.math.reduce_mean(tf.square(pred - batch_targets)))

        loss = tf.math.reduce_mean(tf.square(pred - batch_targets)) + dy_dx_reg*tf.math.reduce_mean(tf.square(dy_dx))
        # print('loss time', time() - t)

    if train:
        loss_grad = model_tape.gradient(loss, model.trainable_variables)
        # print('grad time', time() - t)

        opt.apply_gradients(zip(loss_grad, model.trainable_variables))

    return loss


def run_epoch(model, opt, inputs, targets, train = True, dy_dx_reg = 1e-5):
    batch_losses = []
    n_batches = math.ceil(inputs.shape[0] / batch_size)

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size

        batch_loss = run_batch(model, opt, inputs[start:end], targets[start:end], dy_dx_reg=dy_dx_reg, train = train)
        batch_losses.append(batch_loss)
        # print('opt time', time() - t)

    return batch_losses

def custom_fit(model, inputs, targets, val_prop, dy_dx_reg = 1e-5, verbose=False):

    split = int((1-val_prop)*len(inputs))
    train_inputs = inputs[:split]
    train_targets = targets[:split]

    val_inputs = inputs[split:]
    val_targets = targets[split:]

    opt = keras.optimizers.Adam()

    training_losses = []
    validation_losses = []

    for epoch in range(n_epochs):

        train_losses = run_epoch(model, opt, train_inputs, train_targets, train=True, dy_dx_reg=dy_dx_reg)
        val_losses = run_epoch(model, opt, val_inputs, val_targets, train=False, dy_dx_reg=dy_dx_reg)

        training_losses.append(tf.math.reduce_mean(train_losses))

        validation_losses.append(tf.math.reduce_mean(val_losses))

        if verbose:
            tf.print('epoch:', epoch, training_losses[-1], validation_losses[-1])

    return training_losses, validation_losses, model

if __name__ == '__main__':
    np.random.seed(0)

    num_species = 100
    species_prob = 1
    num_pert = 0
    num_metabolites = 0

    # construct interaction matrix
    zero_prop = 0.7
    known_zero_prop = 0.5

    mu, M, C, ICs = generate_params(num_species, num_pert, hetergeneous=False)


    zeros = np.where(M==0)


    randomize = np.arange(M.shape[0])
    np.random.shuffle(randomize)

    known_zeros = [zeros[0][randomize][:int(len(zeros) * known_zero_prop)],
                   zeros[1][randomize][:int(len(zeros) * known_zero_prop)]]

    # construct growth rates matrix


    # instantiate simulator
    simulator = gMLV_sim(num_species=num_species,
                         num_metabolites=num_metabolites,
                         M=M,
                         mu=mu,
                         C=C)
    #simulator.print()

    num_timecourses = 100
    tmax = 100
    n_epochs = 500
    batch_size = 32
    noise_std = 0.0
    val_prop = 0.1


    if len(sys.argv) == 3:
        exp = int(sys.argv[2]) -1



        tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
        # inestigation scan over

        num_timecoursess = [100, 500, 1000, 5000]
        known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
        #species_probs = [0.1, 0.25, 0.5, 0.75, 1.]
        dy_dx_regs = [100, 10, 1, 0.1, 0.01]

        num_timecourses = num_timecoursess[tc]
        known_zero_prop = known_zero_props[zp]
        #species_prob = species_probs[sp]
        species_prob = 1
        dy_dx_reg = dy_dx_regs[sp]

        save_path = sys.argv[1] + '/repeat' + sys.argv[2] + '/'

        os.makedirs(save_path, exist_ok=True)
    elif len(sys.argv) == 2:
        save_path = sys.argv[1] + '/'
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = './working_dir'

    #n_batch_updates = n_epochs*900/32 # change n_epochs so that the same number of batch updates are run for each test
    #n_epochs = int(32*n_batch_updates/(num_timecourses*0.9))

    print(num_timecourses, known_zero_prop, species_prob, n_epochs)

    transplant_pert = True

    if transplant_pert:
        num_pert = num_species
    sampling_time = 10
    dt = 1

    if transplant_pert:
        ryobs, rysim, all_perts = generate_data_transplant()

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
    inputs[:, 1:, :] = 0 #rmove everything apart from ICs in inputs

    inputs = np.concatenate((inputs, all_perts), axis = 2).astype(np.float32)
    print(inputs.shape)
    targets = copy.deepcopy(ryobs[:,1:,:]).astype(np.float32)


    print(inputs.shape, targets.shape) # (n_simeseries, n_timepoints, n_species)

    ## FIT RNN
    print(len(sampling_times))


    # custom training loop to incorporate prior knowledge
    L2_regs = [1e-8, 1e-7, 1e-6, 1e-5]
    GRU_sizes = [32, 64, 128, 256, 512]
    dy_dx_regs = [1e-2, 1e-3, 1e-4]


    # best parameters from param scan
    L2_reg = 1e-7
    #dy_dx_reg = 0.01
    GRU_size = 256

    print(L2_reg, GRU_size, dy_dx_reg)




    model = get_RNN(num_species, num_pert, len(sampling_times), GRU_size=GRU_size, L2_reg=L2_reg)
    train_loss, val_loss, model = custom_fit(model, inputs, targets, val_prop, dy_dx_reg=dy_dx_reg, verbose=True)

    pred = model.predict(inputs)

    np.save(save_path + '/inputs.npy', inputs)
    np.save(save_path + '/preds.npy', pred)
    np.save(save_path + '/targets.npy', targets)
    np.save(save_path + '/val_loss.npy', val_loss)
    np.save(save_path + '/train_loss.npy', train_loss)
    #model.save(save_path + '/RNN' ) # not working on cluster
    sys.exit()
    #history = model.fit(inputs, targets, verbose = True, batch_size = batch_size, epochs = n_epochs, validation_split=0.1)

    #print(history.history)


    print(pred.shape)
    print(rysim.shape)
    for i in range(10):
        #print(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])))
        #print(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],pred[-i,:,:])))
        #plot_fit_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])), np.vstack((inputs[-i,0,:num_species][np.newaxis,:],pred[-i,:,:])),None, None, times)
        plot_fit_gMLV_pert( np.vstack((inputs[-i,0,:num_species][np.newaxis,:],pred[-i,:,:])), all_perts[-i], None, None, sampling_times, rysim[-i], times)
        #plot_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])),None, times)
        plt.savefig(save_path + '/test_plot_'+str(i) + '.png', dpi = 300)

        plot_fit_gMLV_pert(np.vstack((inputs[i, 0, :num_species][np.newaxis, :], pred[i, :, :])),
                           all_perts[i], None, None, sampling_times, rysim[i], times)
        # plot_gMLV(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])),None, times)
        plt.savefig(save_path + '/train_plot_' + str(i) + '.png', dpi=300)

    plt.figure()
    plt.plot(train_loss, label = 'train')
    plt.plot(val_loss, label = 'test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path + '/train_test_SSE.png', dpi=300)


    plt.show()
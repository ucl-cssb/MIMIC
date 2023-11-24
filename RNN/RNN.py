import tensorflow as tf
#tf.config.run_functions_eagerly(True)
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
import copy
import math
import numpy as np

def get_RNN(num_species, num_pert, num_ts, GRU_size=32, L2_reg = 0., batch_size = 32):
    '''
    initialises and returns an RNN
    '''
    model = keras.Sequential()

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(keras.Input(shape=(num_ts - 1, num_species + num_pert), name="S_input", batch_size = batch_size))

    #model.add(layers.Dense(100, use_bias = False)) # 'embedding' layer

    model.add(layers.GRU(GRU_size, return_sequences=True, unroll = True, stateful = True))

    model.add(layers.GRU(GRU_size, return_sequences=True, unroll = True, stateful = True, kernel_regularizer=regularizers.L2(L2_reg)))

    model.add(layers.Dense(num_species, kernel_regularizer=regularizers.L2(L2_reg)))

    #model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

    return model

@tf.function
def run_batch(model, opt, batch_ab, batch_pert, known_zeros, train = True, dy_dx_reg = 1e-5, batch_size = 32):
    '''
    Runs a batch through the RNN and trains using the custom training loss which regularises the know interaction matrix zeros to be zero
    '''


    with tf.GradientTape(persistent = True) as model_tape:

        # get trainable variables
        train_vars = model.trainable_variables


        abundances = batch_ab[:, 0:1] #get ICs
        all_preds = [abundances]
        total_pred_loss = 0.
        total_reg_loss = 0.

        for i in range(batch_ab.shape[1]-1):

            perts = batch_pert[:, i:i+1]  # get perts for the first timestep

            batch_inputs = tf.concat([abundances, perts], 2)

            with tf.GradientTape(persistent=True) as loss_tape:
                loss_tape.watch(batch_inputs)


                pred = model(batch_inputs)  # (32, 1, 10)
            all_preds.append(pred)

            # first get gradients of pred wrt inputs
            dy_dx = loss_tape.batch_jacobian(pred,
                                             batch_inputs)  #  (batch, time, species, time, species+perts)

            # calculate loss, these derivatives of the model should be zero, because we know these interactins are 0
            dy_dx_loss = 0.
            for n in range(len(known_zeros[0])):
                dy_dx_loss = tf.add(dy_dx_loss, tf.reduce_mean(tf.square(
                    dy_dx[:, 0, known_zeros[1][n], 0, known_zeros[0][n]])))  # [all_batches, 1, n_sp, 1, n_sp]


            targets = batch_ab[:, i+1:i+2] # get the targets for this time point

            total_pred_loss = tf.add(total_pred_loss, tf.math.reduce_mean(tf.square(pred - targets)))
            total_reg_loss = tf.add(total_reg_loss, tf.multiply(dy_dx_reg, dy_dx_loss))

            abundances = pred

        total_loss = tf.divide(tf.add(total_reg_loss, total_pred_loss), batch_size)  #pred and p,oss due to prior kowledge
        #total_loss = tf.divide(total_pred_loss, batch_size) # pred loss only




    if train:


        grad = model_tape.gradient(total_loss, train_vars)

        opt.apply_gradients(zip(grad, train_vars))

    model.reset_states()

    return all_preds,  total_loss

def run_epoch(model, opt, abundances, perturbations, known_zeros, train = True, dy_dx_reg = 1e-5, batch_size = 32):
    '''
    runs one epoch of training
    '''
    batch_losses = []
    n_batches = abundances.shape[0] // batch_size
    all_preds = []
    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size


        preds, batch_loss = run_batch(model, opt, abundances[start:end], perturbations[start:end], known_zeros, dy_dx_reg=dy_dx_reg, train = train, batch_size=batch_size)

        preds = np.squeeze(np.array(preds), 2)

        preds = np.swapaxes(preds, 0, 1)
        all_preds.extend(preds)


        batch_losses.append(batch_loss)
        # print('opt time', time() - t)

    return all_preds, batch_losses

def custom_fit(model, abundances, perturbations, known_zeros, n_epochs, val_prop, dy_dx_reg = 1e-5, verbose=False, batch_size = 32):
    '''
    performs the custom fitting of the RNN with prior knowledge of some of the interaction matrix zeros
    '''

    abundances = abundances.astype(np.float32)  # species levels and perturbations for each time point
    perturbations = perturbations.astype(np.float32)

    split = int((1 - val_prop) * len(abundances) - 1)

    split = batch_size * math.floor(split/batch_size)

    train_ab = abundances[:split]
    train_pert = perturbations[:split]
    val_ab = abundances[split:]
    val_pert = perturbations[split:]

    opt = keras.optimizers.Adam()

    training_losses = []
    validation_losses = []

    for epoch in range(n_epochs):

        train_preds, train_losses = run_epoch(model, opt, train_ab, train_pert, known_zeros, train=True, dy_dx_reg=dy_dx_reg, batch_size=batch_size)
        val_preds, val_losses = run_epoch(model, opt, val_ab, val_pert, known_zeros, train=False, dy_dx_reg=dy_dx_reg, batch_size=batch_size)

        training_losses.append(tf.math.reduce_mean(train_losses))

        validation_losses.append(tf.math.reduce_mean(val_losses))

        if verbose:
            tf.print('epoch:', epoch, training_losses[-1], validation_losses[-1])

    return train_preds, val_preds, training_losses, validation_losses, model
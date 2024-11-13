import numpy as np
import deepxde as dde
from deepxde.backend import tf
from mimic.model_infer import *
from mimic.model_simulate import *

# Define the ODE system for gLV


def gLV_ODE(t, y, params):
    num_species = y.shape[1]
    mu = params[:num_species]
    M = tf.reshape(params[num_species:], (num_species, num_species))

    dy_dt = y * (mu + tf.matmul(y, M, transpose_b=True))
    return [tf.gradients(y[:, i], t)[0] - dy_dt[:, i:i+1] for i in range(num_species)]


# generate some data
num_species = 3
M = np.zeros((num_species, num_species))
np.fill_diagonal(M, [-0.05, -0.1, -0.15])
M[0, 1] = 0.05
M[1, 0] = -0.02

# construct growth rates matrix
# mu = np.random.lognormal(0.01, 0.5, num_species)
mu = np.array([0.8, 1.2, 1.5])

# instantiate simulator
simulator = sim_gLV(num_species=num_species,
                    M=M,
                    mu=mu)
simulator.print_parameters()

init_species = 10 * np.ones(num_species)
t_data = np.arange(0, 10, 0.1)
yobs, y0, mu, M, _ = simulator.simulate(
    times=t_data, init_species=init_species)

y_data = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)


# Time domain for ODEs
geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])

observes = [
    dde.DirichletBC(
        geom,
        lambda X, index=i: y_data[:, index: index + 1],
        lambda _, on_boundary: on_boundary,
        component=i,
    )
    for i in range(y_data.shape[1])
]
# Prepare PDE data
data = dde.data.PDE(geom, gLV_ODE, observes, anchors=t_data)

# Define neural network
net = dde.maps.FNN([1] + [128] * 3 + [y_data.shape[1]],
                   "swish", "Glorot normal")

# Transform features and outputs if needed


def feature_transform(t):
    return tf.concat([t, tf.sin(t), tf.sin(2 * t)], axis=1)


net.apply_feature_transform(feature_transform)

# Model compilation
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss="MSE")


losshistory, train_state = model.train(epochs=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

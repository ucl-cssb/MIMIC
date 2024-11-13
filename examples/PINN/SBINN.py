import numpy as np
import deepxde as dde
from deepxde.backend import tf
from mimic.model_simulate import sim_gLV

# Define the ODE system for gLV


def gLV_ODE(t, y, params):
    num_species = y.shape[1]
    mu = params[:num_species]
    M = tf.reshape(params[num_species:], (num_species, num_species))

    dy_dt = y * (mu + tf.matmul(y, M, transpose_b=True))
    return [tf.gradients(y[:, i], t)[0] - dy_dt[:, i:i+1] for i in range(num_species)]


# Generate data
num_species = 3
M = np.zeros((num_species, num_species))
np.fill_diagonal(M, [-0.05, -0.1, -0.15])
M[0, 1] = 0.05
M[1, 0] = -0.02

mu = np.array([0.8, 1.2, 1.5])

simulator = sim_gLV(num_species=num_species, M=M, mu=mu)
simulator.print_parameters()

init_species = 10 * np.ones(num_species)
t_data = np.arange(0, 10, 0.1)[:, None]  # Ensure t_data is 2D
yobs, y0, mu, M, _ = simulator.simulate(
    times=t_data.ravel(), init_species=init_species)

y_data = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)

# Define geometry
geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])

# Observations
observes = [
    dde.PointSetBC(t_data, y_data[:, i:i+1], component=i) for i in range(num_species)
]

# Prepare PDE data
params_init = np.hstack([mu, M.flatten()])
params = tf.Variable(params_init, trainable=True, dtype=tf.float32)

data = dde.data.PDE(geom, lambda t, y: gLV_ODE(
    t, y, params), observes, anchors=t_data)

# Define neural network
net = dde.maps.FNN([1] + [128] * 3 + [num_species], "swish", "Glorot normal")

# Apply feature transformation


def feature_transform(t):
    return tf.concat([t, tf.sin(t), tf.sin(2 * t)], axis=1)


net.apply_feature_transform(feature_transform)

# Model compilation and training
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss="MSE")

losshistory, train_state = model.train(epochs=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

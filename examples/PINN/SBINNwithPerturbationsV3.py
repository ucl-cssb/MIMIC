import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from mimic.model_simulate import sim_gLV
import tensorflow as tf


# Define the ODE system for gLV with perturbations
def gLV_perturbed_ODE(t, y, params, u_fn=None):
    num_species = y.shape[1]
    num_perturbations = u_fn(0).shape[0]  # Assume u(t) is non-zero for all t
    mu = params[:num_species]
    M = tf.reshape(params[num_species:num_species +
                   num_species**2], (num_species, num_species))
    epsilon = tf.reshape(
        params[num_species + num_species**2:], (num_species, num_perturbations))

    # Ensure u_t has the correct shape
    # Shape: [1, num_perturbations]
    u_t = tf.reshape(u_fn(t), (1, num_perturbations))
    perturbation_effect = tf.matmul(
        u_t, tf.transpose(epsilon))  # Result: [1, num_species]

    # Add perturbation effect to the model
    dy_dt = y * (mu + tf.matmul(y, M, transpose_b=True) +
                 tf.squeeze(perturbation_effect, axis=0))
    return [tf.gradients(y[:, i], t)[0] - dy_dt[:, i:i+1] for i in range(num_species)]


# Define a sample perturbation function
def perturbation_fn(t):
    # Check if input is a TensorFlow tensor
    if isinstance(t, tf.Tensor):
        return tf.stack([tf.sin(0.1 * t), tf.cos(0.1 * t)])
    else:
        # Handle standard numeric input
        return np.array([np.sin(0.1 * t), np.cos(0.1 * t)])


# Simulation Parameters
num_species = 3
num_perturbations = 2
M = np.zeros((num_species, num_species))
np.fill_diagonal(M, [-0.05, -0.1, -0.15])
M[0, 1] = 0.05
M[1, 0] = -0.02
mu = np.array([0.8, 1.2, 1.5])
epsilon = np.random.normal(0, 0.1, size=(num_species, num_perturbations))

simulator = sim_gLV(num_species=num_species,
                    num_perturbations=num_perturbations, M=M, mu=mu, epsilon=epsilon)
simulator.print_parameters()

# Simulate data
init_species = 10 * np.ones(num_species)  # Ensure length matches num_species
times = np.arange(0, 10, 0.1)
yobs, _, mu, M, epsilon = simulator.simulate(
    # u matches num_perturbations
    times=times, init_species=init_species, u=perturbation_fn
)
y_data = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)

# Define geometry and observations
t_data = times[:, None]
geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])
observes = [dde.PointSetBC(t_data, y_data[:, i:i+1], component=i)
            for i in range(num_species)]

# Prepare PDE data
params_init = np.hstack([mu, M.flatten(), epsilon.flatten()])
params = tf.Variable(params_init, trainable=True, dtype=tf.float32)
data = dde.data.PDE(geom, lambda t, y: gLV_perturbed_ODE(
    t, y, params, perturbation_fn), observes, anchors=t_data)

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

# Predict and analyze
y_pred = model.predict(t_data)

# Extract inferred parameters
inferred_params = model.sess.run(params)
inferred_mu = inferred_params[:num_species]
inferred_M = inferred_params[num_species:num_species +
                             num_species**2].reshape(num_species, num_species)
inferred_epsilon = inferred_params[num_species +
                                   num_species**2:].reshape(num_species, num_perturbations)

# Plot species abundances
plt.figure(figsize=(10, 6))
for i in range(num_species):
    plt.plot(t_data, y_data[:, i], "o",
             label=f"Actual (species {i+1})", markersize=4)
    plt.plot(t_data, y_pred[:, i], "-", label=f"Predicted (species {i+1})")
plt.xlabel("Time")
plt.ylabel("Species Abundance")
plt.legend()
plt.title("Predicted vs Actual Data for All Species")
plt.savefig("species_comparison_with_perturbations.png")
plt.show()

# Compare epsilon
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
im1 = axes[0].imshow(epsilon, cmap="viridis", aspect="auto")
axes[0].set_title("Actual Perturbation Matrix (ε)")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(inferred_epsilon, cmap="viridis", aspect="auto")
axes[1].set_title("Inferred Perturbation Matrix (ε)")
plt.colorbar(im2, ax=axes[1])
plt.tight_layout()
plt.savefig("perturbation_matrix_comparison.png")
plt.show()

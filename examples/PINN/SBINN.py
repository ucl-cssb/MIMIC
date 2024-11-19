import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
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

# Predict the solution using the trained model
y_pred = model.predict(t_data)

# Extract inferred parameters from the model
inferred_params = model.sess.run(params)
inferred_mu = inferred_params[:num_species]
inferred_M = inferred_params[num_species:].reshape(num_species, num_species)

# Plot predicted vs actual data for species abundances
for i in range(num_species):
    plt.figure()
    plt.plot(t_data, y_data[:, i], "o",
             label=f"Actual (species {i+1})", markersize=4)
    plt.plot(t_data, y_pred[:, i], "-", label=f"Predicted (species {i+1})")
    plt.xlabel("Time")
    plt.ylabel(f"Species {i+1} Abundance")
    plt.legend()
    plt.title(f"Species {i+1}: Predicted vs Actual")
    plt.savefig(f"species_{i+1}_comparison.png")
    plt.show()

# Plot actual vs inferred M matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Actual M matrix
im1 = axes[0].imshow(M, cmap="viridis", aspect="auto")
axes[0].set_title("Actual Interaction Matrix (M)")
plt.colorbar(im1, ax=axes[0])

# Inferred M matrix
im2 = axes[1].imshow(inferred_M, cmap="viridis", aspect="auto")
axes[1].set_title("Inferred Interaction Matrix (M)")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("interaction_matrix_comparison.png")
plt.show()

# Optional: Print numerical comparison
print("Actual M:\n", M)
print("Inferred M:\n", inferred_M)

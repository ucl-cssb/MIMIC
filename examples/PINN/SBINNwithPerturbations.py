import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from mimic.model_simulate import sim_gLV


# Define known perturbations compatible with TensorFlow
def perturbations_fn(t):
    """
    Example perturbation function (time-dependent).
    Uses TensorFlow operations to handle symbolic tensors.
    """
    return tf.stack([tf.sin(0.1 * t), tf.cos(0.1 * t)], axis=0)

# Define the ODE system with perturbations


def gLV_ODE_with_perturbations(t, y, params, num_species, num_perturbations, perturbations_fn=None):
    """
    ODE system for gLV with perturbations.

    Args:
        t: Time.
        y: Species abundances.
        params: Trainable parameters (growth rates, interaction matrix, perturbation effects).
        num_species: Number of species.
        num_perturbations: Number of perturbations.
        perturbations_fn: Function for known perturbations (u(t)).
    Returns:
        Residuals for the ODE system.
    """
    mu = params[:num_species]
    M = tf.reshape(params[num_species:num_species +
                   num_species**2], (num_species, num_species))
    epsilon = tf.reshape(
        params[num_species + num_species**2:], (num_species, num_perturbations))

    # Compute perturbation effects
    if perturbations_fn is not None:
        u_t = perturbations_fn(t)  # Use TensorFlow-compatible perturbations
    else:
        u_t = tf.zeros((num_perturbations,), dtype=tf.float32)

    dy_dt = y * (mu + tf.matmul(y, M, transpose_b=True) +
                 tf.matmul(epsilon, u_t, transpose_b=True))
    return [tf.gradients(y[:, i], t)[0] - dy_dt[:, i:i+1] for i in range(num_species)]


# Generate synthetic data with perturbations
def generate_synthetic_data(num_species, num_perturbations, t_data, noise_scale=0.1):
    """
    Generates synthetic data using the gLV model with perturbations.

    Args:
        num_species: Number of species.
        num_perturbations: Number of perturbations.
        t_data: Time points for simulation.
        noise_scale: Standard deviation of the noise to add to the simulated data.
    Returns:
        Noisy observed data, true growth rates (mu), true interaction matrix (M), true perturbation effects (epsilon).
    """
    M = np.zeros((num_species, num_species))
    np.fill_diagonal(M, [-0.05 - 0.05 * i for i in range(num_species)])
    epsilon = np.random.uniform(-0.1, 0.1, (num_species, num_perturbations))
    mu = np.linspace(0.8, 1.5, num_species)

    simulator = sim_gLV(num_species=num_species,
                        num_perturbations=num_perturbations, M=M, mu=mu, epsilon=epsilon)

    init_species = 10 * np.ones(num_species)
    yobs, _, _, _, _ = simulator.simulate(
        times=t_data.ravel(), init_species=init_species, u=perturbations_fn)
    y_data = yobs + np.random.normal(loc=0, scale=noise_scale, size=yobs.shape)
    return y_data, mu, M, epsilon


# Train the model
def train_model_with_perturbations(t_data, y_data, num_species, num_perturbations):
    """
    Trains the model for gLV with perturbations.

    Args:
        t_data: Time points for training.
        y_data: Observed noisy data.
        num_species: Number of species.
        num_perturbations: Number of perturbations.
    Returns:
        Trained model, inferred parameters, predictions.
    """
    # Define geometry
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])

    # Observations
    observes = [dde.PointSetBC(t_data, y_data[:, i:i+1], component=i)
                for i in range(num_species)]

    # Initial parameters
    params_init = np.hstack([
        np.random.uniform(0, 1, size=num_species),  # Growth rates
        np.random.uniform(-0.1, 0.1, size=num_species **
                          2),  # Interaction matrix
        np.random.uniform(-0.1, 0.1, size=num_species * \
                          num_perturbations)  # Perturbation effects
    ])
    params = tf.Variable(params_init, trainable=True, dtype=tf.float32)

    # Define PDE data
    data = dde.data.PDE(
        geom,
        lambda t, y: gLV_ODE_with_perturbations(
            t, y, params, num_species, num_perturbations, perturbations_fn),
        observes,
        anchors=t_data,
    )

    # Define neural network
    net = dde.maps.FNN([1] + [128] * 3 + [num_species],
                       "swish", "Glorot normal")

    # Compile and train the model
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss="MSE")
    losshistory, train_state = model.train(epochs=20000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Predictions and inferred parameters
    y_pred = model.predict(t_data)
    inferred_params = model.sess.run(params)
    return model, inferred_params, y_pred


# Visualization
def visualize_results_with_perturbations(t_data, y_data, y_pred, mu, inferred_mu, M, inferred_M, epsilon, inferred_epsilon):
    """
    Visualizes the results for gLV with perturbations.
    """
    # Compare species dynamics
    plt.figure(figsize=(10, 6))
    for i in range(len(mu)):
        plt.plot(t_data, y_data[:, i], "o",
                 label=f"Actual (species {i+1})", markersize=4)
        plt.plot(t_data, y_pred[:, i], "-", label=f"Predicted (species {i+1})")
    plt.xlabel("Time")
    plt.ylabel("Species Abundance")
    plt.legend()
    plt.title("Predicted vs Actual Data for All Species")
    plt.savefig("all_species_comparison.png")
    plt.show()

    # Compare interaction matrix (M)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(M, cmap="viridis", aspect="auto")
    axes[0].set_title("Actual Interaction Matrix (M)")
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(inferred_M, cmap="viridis", aspect="auto")
    axes[1].set_title("Inferred Interaction Matrix (M)")
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.savefig("interaction_matrix_comparison_with_values.png")
    plt.show()

    # Compare perturbation effects (epsilon)
    fig, ax = plt.subplots()
    im = ax.imshow(inferred_epsilon, cmap="coolwarm", aspect="auto")
    ax.set_title("Inferred Perturbation Effects (ε)")
    plt.colorbar(im, ax=ax)
    plt.savefig("inferred_perturbations.png")
    plt.show()


# Main workflow
if __name__ == "__main__":
    num_species = 3
    num_perturbations = 2
    t_data = np.arange(0, 10, 0.1)[:, None]

    # Generate synthetic data
    y_data, mu, M, epsilon = generate_synthetic_data(
        num_species, num_perturbations, t_data)

    # Train the model
    model, inferred_params, y_pred = train_model_with_perturbations(
        t_data, y_data, num_species, num_perturbations)

    # Extract inferred parameters
    inferred_mu = inferred_params[:num_species]
    inferred_M = inferred_params[num_species:num_species +
                                 num_species**2].reshape(num_species, num_species)
    inferred_epsilon = inferred_params[num_species +
                                       num_species**2:].reshape(num_species, num_perturbations)

    # Visualize results
    visualize_results_with_perturbations(
        t_data, y_data, y_pred, mu, inferred_mu, M, inferred_M, epsilon, inferred_epsilon)

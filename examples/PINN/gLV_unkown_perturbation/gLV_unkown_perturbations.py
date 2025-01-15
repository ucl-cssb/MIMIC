"""
Script: sbinnglv_known_perturbations.py

How It Works:

1. Known ϵ:
   We define `epsilon_known` in the script as a fixed vector of shape (num_species,). 
   It is NOT included in `trainable_params`, so the PINN does not try to learn it.

2. Known u(t):
   We define `step_perturbation_tf(t)` as a TensorFlow function returning a vector 
   of shape (None,), indicating which time points are “active” in the batch.
   For simulation (synthetic data), we define the matching NumPy version 
   `step_perturbation_np(t)`.

3. Only μ and M Are Learned:
   We concatenate μ (shape (num_species,)) and M (shape (num_species^2,)) into a 
   single TF variable. In the PDE residual, we parse them out and compute the ODE:

       d y / d t = y * ( mu + M*y + epsilon*u(t) ).

4. No More Errors About Shape:
   Because epsilon is known and passed as a constant (NumPy array or tf.constant), and 
   we carefully broadcast shapes in the PDE, the script avoids the dimension errors 
   seen in earlier attempts.

5. Next Steps:
   Once this script works and recovers μ and M accurately, you can adapt it to:
     - Add unknown epsilon by making it part of the trainable parameters 
       (concatenating it in `trainable_params`).
     - Handle multiple perturbations by adjusting shapes and the PDE residual accordingly.
"""

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf


##############################################################################
# 1) Define a "true" gLV system with a known perturbation
##############################################################################
def step_perturbation_tf(t):
    """
    A step function in TensorFlow:
    Returns 1.0 if t >= 5.0, else 0.0.
    t will be shape (None, 1).
    Output shape (None,).
    """
    val = tf.where(t >= 5.0, tf.ones_like(
        t), tf.zeros_like(t))  # shape (None,1)
    return tf.reshape(val, [-1])  # flatten to shape (None,)


def step_perturbation_np(t):
    """
    The same step function in NumPy for the ODE simulator:
    Returns 1.0 if t >= 5.0, else 0.0.
    t is a scalar float time.
    """
    return 1.0 if t >= 5.0 else 0.0


def true_glv_ode_with_perturbation(N, t, mu, M, epsilon):
    """
    The "true" ODE for forward simulation in NumPy (scipy.integrate.odeint).
    N: array of shape (num_species,)
    t: scalar time
    mu: shape (num_species,)
    M: shape (num_species, num_species)
    epsilon: shape (num_species,)
    returns dN/dt of shape (num_species,)
    """
    u_t = step_perturbation_np(t)  # scalar step function
    # The system: dN/dt = N * (mu + M*N + epsilon*u(t))
    growth = mu + M.dot(N) + epsilon * u_t  # shape (num_species,)
    return N * growth  # element-wise multiply => shape (num_species,)


##############################################################################
# 2) Generate synthetic data (the "ground truth")
##############################################################################
def generate_data(num_species, mu, M, epsilon, t_span, init_species, noise_std=0.05):
    """
    Generate data by integrating the ODE and adding noise.

    num_species: int
    mu: shape (num_species,)
    M: shape (num_species, num_species)
    epsilon: shape (num_species,) - known but not trainable
    t_span: 1D array of time points
    init_species: shape (num_species,)
    noise_std: float standard deviation of Gaussian noise
    """
    from scipy.integrate import odeint

    def ode_wrapper(N, t):
        return true_glv_ode_with_perturbation(N, t, mu, M, epsilon)

    # (len(t_span), num_species)
    sol = odeint(ode_wrapper, init_species, t_span)
    # Add noise
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return sol, noisy_sol


##############################################################################
# 3) Define the PINN ODE/PDE residual function with a known epsilon, known u(t)
##############################################################################
def glv_pde_with_known_perturbation(t, y, trainable_params, num_species, epsilon):
    """
    PDE residual for deepxde, shape logic:

    - y: shape (None, num_species) [the NN output for each time in the batch].
    - trainable_params: 1D tf.Variable, concatenating [mu, M_flat].
      => mu shape = (num_species,)
      => M shape = (num_species, num_species)
    - epsilon is known/fixed. shape = (num_species,).
    - We define a step function u(t) in TF and incorporate it.

    Returns a list of 'num_species' residuals: d/dt of y[:,i] - [some expression].
    """
    # parse mu and M from trainable_params
    mu = trainable_params[:num_species]  # shape (num_species,)
    M_flat = trainable_params[num_species:]
    M_ = tf.reshape(M_flat, (num_species, num_species))

    # shape (None,) => step function
    u_t = step_perturbation_tf(t)

    # Expand for broadcasting
    mu_expanded = tf.expand_dims(mu, axis=0)         # shape (1, num_species)
    My = tf.matmul(y, M_, transpose_b=True)         # shape (None, num_species)

    eps_expanded = tf.expand_dims(epsilon, axis=0)  # shape (1, num_species)
    u_expanded = tf.expand_dims(u_t, axis=1)        # shape (None, 1)
    # => (eps_expanded * u_expanded) => shape (None, num_species)

    # growth = mu + M*y + epsilon*u(t)
    growth = mu_expanded + My + (eps_expanded * u_expanded)
    dy_dt = y * growth  # shape (None, num_species)

    # PDE residual: d/dt of y[:, i] - dy_dt[:, i] = 0
    residuals = []
    for i in range(num_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]           # shape (None,)
        res_i = dyi_dt - dy_dt[:, i: i + 1]           # shape (None,1)
        residuals.append(res_i)
    return residuals


##############################################################################
# 4) (OPTIONAL) Helper function to compare parameters
##############################################################################
def compare_params(mu=None, M=None):
    """
    A helper function to produce stem plots (or any other comparative plot)
    for actual vs. inferred mu and M. This is borrowed from your previous scripts.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if mu is not None:
        fig, ax = plt.subplots()
        # mu is a tuple (mu_actual, mu_inferred)
        mu_actual, mu_inferred = mu
        ax.stem(np.arange(0, len(mu_actual)), mu_actual, markerfmt="X",
                label='Actual μ', linefmt='C1-')
        ax.stem(np.arange(0, len(mu_inferred)), mu_inferred, markerfmt="D",
                label='Inferred μ', linefmt='C0-')
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('μ[i]')
        ax.legend()
        plt.title("Comparison of Actual vs Inferred Growth Rates (μ)")
        plt.savefig("mu_comparison.png")
        plt.show()

    if M is not None:
        fig, ax = plt.subplots()
        # M is a tuple (M_actual, M_inferred)
        M_actual, M_inferred = M
        ax.stem(np.arange(0, M_actual.size), M_actual.flatten(),
                markerfmt="X", label='Actual M', linefmt='C1-')
        ax.stem(np.arange(0, M_inferred.size), M_inferred.flatten(),
                markerfmt="D", label='Inferred M', linefmt='C0-')
        ax.set_xlabel('Matrix Element Index')
        ax.set_ylabel('M[i, j]')
        ax.legend()
        plt.title("Comparison of Actual vs Inferred Interaction Matrix (M)")
        plt.savefig("M_comparison.png")
        plt.show()


##############################################################################
# 5) Main script: set up PINN, train, compare results
##############################################################################
def main():
    # --- a) Choose system size and "true" parameters
    num_species = 3
    mu_true = np.array([0.8, 1.2, 1.5], dtype=np.float32)
    M_true = np.zeros((num_species, num_species), dtype=np.float32)
    np.fill_diagonal(M_true, [-0.05, -0.1, -0.15])
    M_true[0, 1] = 0.05
    M_true[1, 0] = -0.02
    # Known epsilon (not trainable)
    epsilon_known = np.array([-0.1, 0.2, -0.05], dtype=np.float32)

    # --- b) Generate synthetic data
    t_span = np.arange(0, 10, 0.1, dtype=np.float32)
    init_species = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    # Return the "clean" solution and the noisy version
    _, y_data = generate_data(
        num_species, mu_true, M_true, epsilon_known, t_span, init_species, noise_std=0.1
    )

    t_data = t_span.reshape(-1, 1)  # shape (N,1)

    # --- c) Build PDE data in DeepXDE
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])

    # Observations: y(t_data) = y_data
    observes = [
        dde.PointSetBC(t_data, y_data[:, i: i + 1], component=i)
        for i in range(num_species)
    ]

    # We only train [mu, M], not epsilon
    param_init = np.hstack([
        np.random.uniform(0.5, 1.5, size=num_species),      # mu guess
        np.random.uniform(-0.2, 0.2, size=(num_species**2,))  # M guess
    ]).astype(np.float32)
    trainable_params = tf.Variable(
        param_init, trainable=True, dtype=tf.float32)

    def ode_residual(t, y):
        return glv_pde_with_known_perturbation(
            t, y, trainable_params, num_species, epsilon_known
        )

    data = dde.data.PDE(
        geom,
        ode_residual,
        observes,
        anchors=t_data,  # PDE anchor points
    )

    # --- d) Define the network for y(t)
    net = dde.maps.FNN(
        [1] + [128, 128, 128] + [num_species],
        activation="swish",
        kernel_initializer="Glorot normal",  # or "glorot_uniform"
        # You can also specify bias_initializer if you like
    )

    # Optional feature transform
    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)

    net.apply_feature_transform(feature_transform)

    # Build and train model
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss="MSE")

    losshistory, train_state = model.train(epochs=20000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # --- e) Inference: get the learned mu, M
    inferred = model.sess.run(trainable_params)
    num_params_mu = num_species
    num_params_M = num_species * num_species
    mu_learned = inferred[:num_params_mu]
    M_learned = inferred[num_params_mu: (num_params_mu + num_params_M)].reshape(
        (num_species, num_species)
    )

    # --- f) Predict solution on the same t_data
    y_pred = model.predict(t_data)

    # --- g) Compare results in console
    print("======== True vs Learned Parameters ========")
    print("True mu = ", mu_true)
    print("Learned mu = ", mu_learned)
    print("")
    print("True M = \n", M_true)
    print("Learned M = \n", M_learned)
    print("")
    print("Known epsilon = ", epsilon_known, " (not trained)")

    # --- h) Plot data vs PINN predictions
    plt.figure(figsize=(10, 6))
    for i in range(num_species):
        plt.plot(t_data, y_data[:, i], "o",
                 label=f"Observed S{i+1}", markersize=4)
        plt.plot(t_data, y_pred[:, i], "-", label=f"PINN S{i+1}")
    plt.xlabel("Time")
    plt.ylabel("Species Abundance")
    plt.title("gLV with Known Perturbation: Observations vs PINN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pinn_known_perturbation_solution.png")
    plt.show()

    # --- i) Compare M via a heatmap (actual vs inferred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(M_true, cmap="viridis", aspect="auto")
    axes[0].set_title("Actual Interaction Matrix (M)")
    plt.colorbar(im1, ax=axes[0])
    for (rowi, coli), val in np.ndenumerate(M_true):
        axes[0].text(coli, rowi, f"{val:.2f}",
                     ha="center", va="center", color="w")

    im2 = axes[1].imshow(M_learned, cmap="viridis", aspect="auto")
    axes[1].set_title("Inferred Interaction Matrix (M)")
    plt.colorbar(im2, ax=axes[1])
    for (rowi, coli), val in np.ndenumerate(M_learned):
        axes[1].text(coli, rowi, f"{val:.2f}",
                     ha="center", va="center", color="w")

    plt.tight_layout()
    plt.savefig("interaction_matrix_comparison_with_values.png")
    plt.show()

    # --- j) Use our compare_params helper for stem plots
    compare_params(mu=(mu_true, mu_learned), M=(M_true, M_learned))


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

import deepxde as dde
from deepxde.backend import tf


def step_perturbation_tf(t):
    """
    A step function in TensorFlow. Returns 1.0 if t >= 5.0, else 0.0.
    t shape: (None, 1)
    Output shape: (None,)
    """
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def step_perturbation_np(t):
    """
    The same step function in NumPy for ODE simulation.
    Returns 1.0 if t >= 5.0, else 0.0.
    """
    return 1.0 if t >= 5.0 else 0.0


def true_glv_ode_with_perturbation(N, t, mu, M, epsilon):
    """
    True ODE for forward simulation in NumPy.  N shape: (num_species,)
    mu shape: (num_species,), M shape: (num_species, num_species), epsilon shape: (num_species,)
    """
    u_t = step_perturbation_np(t)
    growth = mu + M.dot(N) + epsilon * u_t
    return N * growth


def generate_data(num_species, mu, M, epsilon, t_span, init_species, noise_std=0.05):
    """
    Generate synthetic data by integrating the ODE and adding noise.
    """
    from scipy.integrate import odeint

    def ode_wrapper(N, t):
        return true_glv_ode_with_perturbation(N, t, mu, M, epsilon)

    sol = odeint(ode_wrapper, init_species, t_span)
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return sol, noisy_sol


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species):
    """
    PDE residual for unknown epsilon. trainable_params now includes [mu, M_flat, epsilon].
    """
    # parse mu, M, epsilon
    num_params_mu = num_species
    num_params_M = num_species * num_species
    # shape (num_species,)
    mu = trainable_params[:num_params_mu]
    M_flat = trainable_params[num_params_mu: num_params_mu + num_params_M]
    epsilon = trainable_params[num_params_mu +
                               num_params_M:]         # shape (num_species,)

    M = tf.reshape(M_flat, (num_species, num_species))

    # shape (None,)
    u_t = step_perturbation_tf(t)
    # shape (1, num_species)
    mu_expanded = tf.expand_dims(mu, axis=0)
    # shape (None, num_species)
    My = tf.matmul(y, M, transpose_b=True)

    # shape (1, num_species)
    eps_expanded = tf.expand_dims(epsilon, axis=0)
    # shape (None, 1)
    u_expanded = tf.expand_dims(u_t, axis=1)

    growth = mu_expanded + My + eps_expanded * \
        u_expanded             # shape (None, num_species)
    dy_dt = y * growth

    residuals = []
    for i in range(num_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]
        residuals.append(dyi_dt - dy_dt[:, i:i+1])
    return residuals


def compare_params(mu=None, M=None, epsilon=None, prefix=""):
    """
    Helper function to stem-plot and compare actual vs inferred parameters
    for a gLV model (or similar).

    Arguments:
        mu: A tuple (mu_actual, mu_inferred) if comparing growth rates μ.
            Each must be a 1D numpy array.
            If None, it is skipped.
        M: A tuple (M_actual, M_inferred) if comparing interaction matrices.
            Each must be a 2D numpy array of the same shape.
            If None, it is skipped.
        epsilon: A tuple (epsilon_actual, epsilon_inferred) if comparing
                 perturbation effects or any other 1D parameter set.
                 If None, it is skipped.
        prefix: Optional string prefix to prepend to the saved figure filenames.
                For example, prefix="trial1_" will save "trial1_mu_comparison.png",
                etc.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Compare μ (growth rates)
    if mu is not None:
        mu_actual, mu_inferred = mu
        fig, ax = plt.subplots()
        ax.stem(
            np.arange(len(mu_actual)), mu_actual,
            markerfmt="X", label="Actual μ", linefmt="C1-"
        )
        ax.stem(
            np.arange(len(mu_inferred)), mu_inferred,
            markerfmt="D", label="Inferred μ", linefmt="C0-"
        )
        ax.set_xlabel("Species Index")
        ax.set_ylabel("Growth Rate μ[i]")
        ax.legend()
        plt.title("Comparison of Actual vs Inferred Growth Rates (μ)")
        plt.tight_layout()
        plt.savefig(f"{prefix}mu_comparison.png")
        plt.show()

    # Compare M (interaction matrix)
    if M is not None:
        M_actual, M_inferred = M
        fig, ax = plt.subplots()
        ax.stem(
            np.arange(M_actual.size), M_actual.flatten(),
            markerfmt="X", label="Actual M", linefmt="C1-"
        )
        ax.stem(
            np.arange(M_inferred.size), M_inferred.flatten(),
            markerfmt="D", label="Inferred M", linefmt="C0-"
        )
        ax.set_xlabel("Matrix Element Index")
        ax.set_ylabel("M[i,j]")
        ax.legend()
        plt.title("Comparison of Actual vs Inferred Interaction Matrix (M)")
        plt.tight_layout()
        plt.savefig(f"{prefix}M_comparison.png")
        plt.show()

    # Compare ε (perturbation effects or other 1D parameter)
    if epsilon is not None:
        eps_actual, eps_inferred = epsilon
        fig, ax = plt.subplots()
        ax.stem(
            np.arange(len(eps_actual)), eps_actual,
            markerfmt="X", label="Actual ε", linefmt="C1-"
        )
        ax.stem(
            np.arange(len(eps_inferred)), eps_inferred,
            markerfmt="D", label="Inferred ε", linefmt="C0-"
        )
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("ε[i]")
        ax.legend()
        plt.title("Comparison of Actual vs Inferred Perturbation Effects (ε)")
        plt.tight_layout()
        plt.savefig(f"{prefix}epsilon_comparison.png")
        plt.show()


def main():
    num_species = 3
    # "True" parameters for synthetic data
    mu_true = np.array([0.8, 1.2, 1.5], dtype=np.float32)
    M_true = np.zeros((num_species, num_species), dtype=np.float32)
    np.fill_diagonal(M_true, [-0.05, -0.1, -0.15])
    M_true[0, 1] = 0.05
    M_true[1, 0] = -0.02
    epsilon_true = np.array([-0.1, 0.2, -0.05], dtype=np.float32)

    t_span = np.arange(0, 10, 0.1).astype(np.float32)
    init_species = np.array([10.0, 10.0, 10.0], dtype=np.float32)

    # Generate synthetic noisy data
    _, y_noisy = generate_data(
        num_species, mu_true, M_true, epsilon_true, t_span, init_species, noise_std=0.1
    )
    t_data = t_span.reshape(-1, 1)

    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])

    observes = [
        dde.PointSetBC(t_data, y_noisy[:, i: i + 1], component=i)
        for i in range(num_species)
    ]

    # Build trainable params: [mu, M, epsilon]
    param_init = np.hstack(
        [
            # mu guess
            np.random.uniform(0.5, 1.5, size=num_species),
            np.random.uniform(-0.2, 0.2, size=(num_species**2,)
                              ),       # M guess
            # epsilon guess
            np.random.uniform(-0.1, 0.1, size=num_species),
        ]
    ).astype(np.float32)
    trainable_params = tf.Variable(
        param_init, trainable=True, dtype=tf.float32)

    def ode_residual(t, y):
        return glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species)

    data = dde.data.PDE(geom, ode_residual, observes, anchors=t_data)

    net = dde.maps.FNN([1] + [128, 128, 128] + [num_species],
                       activation="swish", kernel_initializer="Glorot normal")

    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)

    net.apply_feature_transform(feature_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss="MSE")
    losshistory, train_state = model.train(epochs=20000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Extract learned parameters
    inferred = model.sess.run(trainable_params)
    num_params_mu = num_species
    num_params_M = num_species * num_species

    mu_learned = inferred[:num_params_mu]
    M_learned = inferred[num_params_mu: num_params_mu +
                         num_params_M].reshape((num_species, num_species))
    epsilon_learned = inferred[num_params_mu + num_params_M:]

    # Predict solution
    y_pred = model.predict(t_data)

    # Print comparisons
    print("======== True vs Learned Parameters ========")
    print("True mu:", mu_true)
    print("Learned mu:", mu_learned)
    print("")
    print("True M:\n", M_true)
    print("Learned M:\n", M_learned)
    print("")
    print("True epsilon:", epsilon_true)
    print("Learned epsilon:", epsilon_learned)

    # Plot solution
    plt.figure(figsize=(10, 6))
    for i in range(num_species):
        plt.plot(t_span, y_noisy[:, i], "o",
                 label=f"Observed S{i+1}", markersize=4)
        plt.plot(t_data, y_pred[:, i], "-", label=f"PINN S{i+1}")
    plt.xlabel("Time")
    plt.ylabel("Species Abundance")
    plt.title("gLV with Unknown Perturbation: Observations vs PINN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pinn_unknown_perturbation_solution.png")
    plt.show()

    compare_params(
        mu=(mu_true, mu_learned),
        M=(M_true, M_learned),
        epsilon=(epsilon_true, epsilon_learned),
        prefix="unknown_perturbation_"
    )


if __name__ == "__main__":
    main()
    # After main finishes, call plot_comparison with actual and inferred values

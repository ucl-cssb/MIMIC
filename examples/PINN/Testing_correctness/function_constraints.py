import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

# Define scaling factors for M and ε.
s_M = 10.0    # Interaction matrix scaling factor
s_eps = 5.0   # Perturbation scaling factor

# Hyperparameters:
# Averaged tuned hyperparameters (from your grid search)
lambda_pde = 1.0      # averaged PDE loss weight
reg_coeff = 0.0046    # averaged L2 regularization coefficient
lr = 0.00075          # averaged learning rate


def step_perturbation_tf(t):
    # TensorFlow step function: returns 1.0 when t>=5, else 0.
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, lambda_pde):
    """
    PDE residual for gLV with unknown parameters.
    The trainable parameters are reparameterized as:
      - φ_mu (length=num_species), with μ = softplus(φ_mu) ensuring μ > 0.
      - φ_M (length=num_species^2). We reshape φ_M into a matrix.
          For off-diagonal entries, we leave them unconstrained.
          For diagonal entries, we set M_ii = -softplus(φ_M_ii) so that they are always negative.
      - eps_scaled (for ε) remains unconstrained.
    We then unscale M and ε by dividing by s_M and s_eps, respectively.
    Each PDE residual is weighted by lambda_pde.
    """
    num_params_mu = num_species
    num_params_M = num_species * num_species

    # Reconstruct μ using softplus:
    phi_mu = trainable_params[:num_params_mu]
    mu = tf.math.softplus(phi_mu)

    # Reconstruct M:
    phi_M = trainable_params[num_params_mu: num_params_mu + num_params_M]
    phi_M_matrix = tf.reshape(phi_M, (num_species, num_species))
    # Create a boolean mask for the diagonal:
    diag_mask = tf.eye(num_species, dtype=tf.bool)
    # For diagonal entries, set M_ii = -softplus(phi_M_ii); for off-diagonals, keep as is.
    M_matrix = tf.where(
        diag_mask, -tf.math.softplus(phi_M_matrix), phi_M_matrix)
    # Unscale M:
    M = M_matrix / s_M

    # For ε, no constraint:
    eps_scaled = trainable_params[num_params_mu + num_params_M:]
    epsilon = eps_scaled / s_eps

    u_t = step_perturbation_tf(t)
    mu_expanded = tf.expand_dims(mu, axis=0)           # shape (1, num_species)
    # shape (None, num_species)
    My = tf.matmul(y, M, transpose_b=True)
    eps_expanded = tf.expand_dims(
        epsilon, axis=0)       # shape (1, num_species)
    u_expanded = tf.expand_dims(u_t, axis=1)             # shape (None, 1)

    growth = mu_expanded + My + eps_expanded * u_expanded
    dy_dt = y * growth

    residuals = []
    for i in range(num_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]
        residuals.append(lambda_pde * (dyi_dt - dy_dt[:, i:i+1]))
    return residuals


def build_feature_transform():
    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return feature_transform


def inverse_softplus(y):
    # Compute an approximate inverse of softplus: x = ln(exp(y) - 1)
    return np.log(np.exp(y) - 1.0)


def infer_parameters_from_file(sim_file):
    """
    Sets up and trains a PINN for parameter inference from simulation data.
    The parameters to be inferred are:
       - μ (intrinsic growth rates), reparameterized as μ = softplus(φ_mu) to enforce μ > 0.
       - M (interaction matrix). The diagonal entries are enforced to be negative via
         M_ii = -softplus(φ_M_ii).
       - ε (perturbation terms) remain unconstrained.
    """
    # Load simulation data
    with open(sim_file, "r") as f:
        sim_data = json.load(f)

    t_span = np.array(sim_data["t_span"], dtype=np.float32)
    t_data = t_span.reshape(-1, 1)
    noisy_solution = np.array(sim_data["noisy_solution"], dtype=np.float32)
    true_mu = np.array(sim_data["mu_true"], dtype=np.float32)
    true_M = np.array(sim_data["M_true"], dtype=np.float32)
    true_epsilon = np.array(sim_data["epsilon_true"], dtype=np.float32)

    num_species = noisy_solution.shape[1]
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])
    observes = [dde.PointSetBC(t_data, noisy_solution[:, i:i+1], component=i)
                for i in range(num_species)]

    # Parameter initialization with constraint transformations:
    num_params_mu = num_species
    num_params_M = num_species * num_species

    # For μ: choose an initial guess in [0.8, 1.6] and invert softplus.
    mu_guess = np.random.uniform(0.8, 1.6, size=num_species)
    phi_mu_guess = inverse_softplus(mu_guess)

    # For M: for off-diagonals, we leave them unconstrained.
    # For diagonal entries, we want M_ii = -softplus(phi) to be between -0.15 and -0.05 * s_M.
    M_guess = np.random.uniform(-0.03, 0.06, size=(num_params_M,))
    for i in range(num_species):
        init_diag = np.random.uniform(-0.15, -0.05) * s_M
        # We need phi such that -softplus(phi) = init_diag; i.e. softplus(phi) = -init_diag.
        phi_M_diag = inverse_softplus(-init_diag)
        M_guess[i * num_species + i] = phi_M_diag

    # For ε, no constraint:
    eps_guess = np.random.uniform(-0.15, 0.25, size=num_species) * s_eps

    param_init = np.hstack(
        [phi_mu_guess, M_guess, eps_guess]).astype(np.float32)
    trainable_params = tf.Variable(
        param_init, trainable=True, dtype=tf.float32)

    def ode_residual(t, y):
        return glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, lambda_pde)

    data = dde.data.PDE(geom, ode_residual, observes, anchors=t_data)

    net = dde.maps.FNN([1] + [128, 128, 128] + [num_species],
                       activation="swish",
                       kernel_initializer="Glorot normal",
                       regularization=["l2", reg_coeff])
    net.apply_feature_transform(build_feature_transform())

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss="MSE")
    losshistory, train_state = model.train(
        iterations=20000, display_every=1000)

    inferred = model.sess.run(trainable_params)
    inferred_phi_mu = inferred[:num_params_mu]
    inferred_phi_M = inferred[num_params_mu: num_params_mu + num_params_M]
    inferred_eps_scaled = inferred[num_params_mu + num_params_M:]
    # Recover μ as softplus(inferred_phi_mu)
    inferred_mu = tf.math.softplus(inferred_phi_mu).eval(session=model.sess)
    # Recover M by applying our transformation:
    inferred_phi_M_matrix = inferred_phi_M.reshape((num_species, num_species))
    # Diagonal: M_ii = -softplus(phi), off-diagonals: unchanged
    diag_mask = np.eye(num_species, dtype=bool)
    inferred_M_matrix = np.where(diag_mask, -tf.math.softplus(
        inferred_phi_M_matrix).eval(session=model.sess), inferred_phi_M_matrix)
    inferred_M = inferred_M_matrix / s_M
    inferred_epsilon = inferred_eps_scaled / s_eps

    y_pred = model.predict(t_data)
    plt.figure(figsize=(10, 6))
    for i in range(num_species):
        plt.plot(t_span, noisy_solution[:, i], "o",
                 label=f"Observed S{i+1}", markersize=3)
        plt.plot(t_data, y_pred[:, i], "-", label=f"PINN S{i+1}")
    plt.xlabel("Time")
    plt.ylabel("Species Abundance")
    plt.title(f"Simulation: {os.path.basename(sim_file)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("inference_results/" +
                os.path.basename(sim_file).replace(".json", "_solution.png"))
    plt.close()

    result = {
        "simulation_file": os.path.basename(sim_file),
        "true_mu": true_mu.tolist(),
        "inferred_mu": inferred_mu.tolist(),
        "true_M": true_M.tolist(),
        "inferred_M": inferred_M.tolist(),
        "true_epsilon": true_epsilon.tolist(),
        "inferred_epsilon": inferred_epsilon.tolist()
    }
    return result


def main():
    sim_dir = "simulations"
    out_dir = "inference_results"
    os.makedirs(out_dir, exist_ok=True)
    sim_files = [os.path.join(sim_dir, f)
                 for f in os.listdir(sim_dir) if f.endswith(".json")]
    all_results = []
    for sim_file in sim_files:
        print(f"Processing {sim_file} ...")
        result = infer_parameters_from_file(sim_file)
        all_results.append(result)
        out_file = os.path.join(out_dir, os.path.basename(
            sim_file).replace(".json", "_inferred.json"))
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved inferred parameters to {out_file}")
    combined_file = os.path.join(out_dir, "all_inferred_parameters.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved all inferred parameters to {combined_file}")


if __name__ == "__main__":
    main()

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
lambda_pde = 1.0      # PDE loss weight
reg_coeff = 0.0046    # L2 regularization coefficient
lr = 0.00075          # Learning rate


def step_perturbation_tf(t):
    # TensorFlow step function: returns 1.0 when t>=5, else 0.
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, lambda_pde):
    """
    PDE residual for gLV with unknown parameters.
    trainable_params: concatenated [mu, M_scaled, eps_scaled].
    M_scaled = M * s_M and eps_scaled = epsilon * s_eps.
    The residuals are multiplied by lambda_pde.
    """
    num_params_mu = num_species
    num_params_M = num_species * num_species
    mu = trainable_params[:num_params_mu]
    M_scaled = trainable_params[num_params_mu: num_params_mu + num_params_M]
    eps_scaled = trainable_params[num_params_mu + num_params_M:]
    # Unscale parameters:
    M = tf.reshape(M_scaled, (num_species, num_species)) / s_M
    epsilon = eps_scaled / s_eps

    u_t = step_perturbation_tf(t)
    mu_expanded = tf.expand_dims(mu, axis=0)
    My = tf.matmul(y, M, transpose_b=True)
    eps_expanded = tf.expand_dims(epsilon, axis=0)
    u_expanded = tf.expand_dims(u_t, axis=1)

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


def create_data(t_data, noisy_solution):
    # Create the data object with current anchors.
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])
    observes = [dde.PointSetBC(t_data, noisy_solution[:, i:i+1], component=i)
                for i in range(noisy_solution.shape[1])]
    return geom, observes


def adaptive_collocation(model, ode_residual, t_min, t_max, num_candidates=1000, num_select=200):
    """
    Evaluate PDE residuals on a dense set of candidate points, then select a subset
    of points with the highest residual error.
    """
    # Generate dense candidate points.
    t_candidates = np.linspace(
        t_min, t_max, num_candidates).reshape(-1, 1).astype(np.float32)
    # Try a vectorized evaluation first. If your ode_residual returns a list of residuals,
    # they can be concatenated along the last axis.
    try:
        residuals = model.predict(t_candidates, operator=ode_residual)
        # If residuals is a list of arrays, combine them.
        if isinstance(residuals, list):
            residuals = np.concatenate([r for r in residuals], axis=1)
        # Compute L2 norm across components for each candidate.
        error_vals = np.linalg.norm(residuals, axis=1)
    except Exception:
        # Fallback: compute residuals point-by-point (less efficient).
        error_vals = []
        for t_val in t_candidates:
            t_val = t_val.reshape(1, 1)
            # Compute residuals at this point.
            residual_list = ode_residual(t_val, model.predict(t_val))
            # If the returned residuals are tensors, convert them to numpy arrays.
            residual_list = [r.numpy() if hasattr(r, "numpy")
                             else np.array(r) for r in residual_list]
            # Combine the residuals (assuming each residual is of shape (1,1)).
            res = np.concatenate([r.reshape(1, -1)
                                 for r in residual_list], axis=1)
            error_vals.append(np.linalg.norm(res))
        error_vals = np.array(error_vals)
    # Select candidate points with the highest error.
    idx = np.argsort(error_vals)[-num_select:]
    return t_candidates[idx]


def infer_parameters_from_file(sim_file):
    """
    Given a simulation JSON file, set up the PINN problem, train with adaptive collocation,
    and return both inferred and true parameters.
    """
    with open(sim_file, "r") as f:
        sim_data = json.load(f)

    t_span = np.array(sim_data["t_span"], dtype=np.float32)
    t_data = t_span.reshape(-1, 1)
    noisy_solution = np.array(sim_data["noisy_solution"], dtype=np.float32)
    true_mu = np.array(sim_data["mu_true"], dtype=np.float32)
    true_M = np.array(sim_data["M_true"], dtype=np.float32)
    true_epsilon = np.array(sim_data["epsilon_true"], dtype=np.float32)
    num_species = noisy_solution.shape[1]

    geom, observes = create_data(t_data, noisy_solution)

    num_params_mu = num_species
    num_params_M = num_species * num_species
    mu_guess = np.random.uniform(0.8, 1.6, size=num_species)
    M_guess = np.random.uniform(-0.03, 0.06, size=(num_params_M,))
    for i in range(num_species):
        M_guess[i * num_species + i] = np.random.uniform(-0.15, -0.05) * s_M
    eps_guess = np.random.uniform(-0.15, 0.25, size=num_species) * s_eps
    param_init = np.hstack([mu_guess, M_guess, eps_guess]).astype(np.float32)
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

    # === Stage 1: Preliminary Training ===
    model.train(iterations=10000, display_every=1000)

    # === Stage 2: Adaptive Collocation ===
    # Evaluate PDE residual on a dense set of candidate points and select those with highest error.
    new_anchors = adaptive_collocation(model, ode_residual, t_data[0, 0], t_data[-1, 0],
                                       num_candidates=1000, num_select=200)
    # Combine new anchors with original ones.
    updated_anchors = np.vstack([t_data, new_anchors])
    # Create a new data object with the updated anchors.
    data_new = dde.data.PDE(geom, ode_residual, observes,
                            anchors=updated_anchors)
    model_new = dde.Model(data_new, net)
    model_new.compile("adam", lr=lr, loss="MSE")
    # Continue training with the new anchors.
    model_new.train(iterations=10000, display_every=1000)

    # Extract inferred parameters.
    inferred = model_new.sess.run(trainable_params)
    inferred_mu = inferred[:num_params_mu]
    inferred_M_scaled = inferred[num_params_mu: num_params_mu + num_params_M]
    inferred_eps_scaled = inferred[num_params_mu + num_params_M:]
    inferred_M = inferred_M_scaled.reshape((num_species, num_species)) / s_M
    inferred_epsilon = inferred_eps_scaled / s_eps

    # (Optional) Plot observed vs. predicted solution.
    y_pred = model_new.predict(t_data)
    plt.figure(figsize=(10, 6))
    for i in range(num_species):
        plt.plot(t_span, noisy_solution[:, i], "o",
                 label=f"Observed S{i+1}", markersize=3)
        plt.plot(t_data, y_pred[:, i], "-", label=f"PINN S{i+1}")
    plt.xlabel("Time")
    plt.ylabel("Species Abundance")
    plt.title(f"Adaptive Collocation: {os.path.basename(sim_file)}")
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

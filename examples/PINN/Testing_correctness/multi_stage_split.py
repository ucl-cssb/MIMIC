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
lambda_pde = 1.0      # Final PDE loss weight (used in the PDE residual)
reg_coeff = 0.0046    # L2 regularization coefficient
lr = 0.00075          # Learning rate

# Supervised loss weights:
lambda_data = 1.0     # Weight for scattered (data) observations
lambda_aux = 1.0      # Weight for auxiliary observations

# Training schedule:
initial_iterations = 1000     # Stage 1: supervised-only training
total_iterations = 20000      # Total training iterations
remaining_iterations = total_iterations - \
    initial_iterations  # Stage 2 iterations
num_ramp_steps = 10           # Number of mini-phases in Stage 2


def step_perturbation_tf(t):
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, current_lambda_pde):
    num_params_mu = num_species
    num_params_M = num_species * num_species
    mu = trainable_params[:num_params_mu]
    M_scaled = trainable_params[num_params_mu: num_params_mu + num_params_M]
    eps_scaled = trainable_params[num_params_mu + num_params_M:]
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
        residuals.append(current_lambda_pde * (dyi_dt - dy_dt[:, i:i+1]))
    return residuals


def build_feature_transform():
    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return feature_transform


def infer_parameters_from_file(sim_file):
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
    # For auxiliary conditions, try using more representative points (e.g., 25% and 75%)
    idx_aux = [len(t_data)//4, 3*len(t_data)//4]
    aux_bcs = []
    for i in range(num_species):
        for idx in idx_aux:
            aux_bcs.append(dde.PointSetBC(
                t_data[idx:idx+1, :], noisy_solution[idx:idx+1, i:i+1], component=i))
    bcs = observes + aux_bcs

    # We use a closure that captures a global variable current_lambda_pde.
    # We start with PDE weight = 0 in Stage 1.
    current_lambda_pde = 0.0  # Initialize here

    def ode_residual(t, y):
        nonlocal current_lambda_pde
        return glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, current_lambda_pde)

    data = dde.data.PDE(geom, ode_residual, bcs, anchors=t_data)

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

    net = dde.maps.FNN([1] + [128, 128, 128] + [num_species],
                       activation="swish",
                       kernel_initializer="Glorot normal",
                       regularization=["l2", reg_coeff])
    net.apply_feature_transform(build_feature_transform())
    model = dde.Model(data, net)

    # Loss components: PDE residuals (num_species) + BC losses (observes: num_species, aux_bcs: 2*num_species) + 1 extra = 4*num_species+1.
    extra_loss_weight = lambda_data
    loss_weights = ([0]*num_species +
                    [lambda_data]*len(observes) +
                    [lambda_aux]*len(aux_bcs) +
                    [extra_loss_weight])
    print("Starting Stage 1 (supervised training only)...")
    model.compile("adam", lr=lr, loss="MSE", loss_weights=loss_weights)
    model.train(iterations=initial_iterations, display_every=1000)

    # Stage 2: Gradually ramp up the PDE loss weight from 0 to 1 over num_ramp_steps mini-phases.
    ramp_iterations = remaining_iterations // num_ramp_steps
    for step in range(num_ramp_steps):
        # gradually increase PDE weight
        current_lambda_pde = (step + 1) / num_ramp_steps
        new_loss_weights = ([current_lambda_pde]*num_species +
                            [lambda_data]*len(observes) +
                            [lambda_aux]*len(aux_bcs) +
                            [extra_loss_weight])
        model.data.loss_weights = new_loss_weights
        print(
            f"Stage 2 ramp step {step+1}/{num_ramp_steps}: setting PDE weight to {current_lambda_pde}")
        model.train(iterations=ramp_iterations, display_every=1000)

    inferred = model.sess.run(trainable_params)
    inferred_mu = inferred[:num_params_mu]
    inferred_M_scaled = inferred[num_params_mu: num_params_mu + num_params_M]
    inferred_eps_scaled = inferred[num_params_mu + num_params_M:]
    inferred_M = inferred_M_scaled.reshape((num_species, num_species)) / s_M
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

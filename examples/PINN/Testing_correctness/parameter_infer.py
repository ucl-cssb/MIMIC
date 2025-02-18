import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

# --- Functions from your PINN inference script ---


def step_perturbation_tf(t):
    # TensorFlow step function: returns 1.0 when t>=5, else 0.
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species):
    """
    PDE residual for gLV with unknown parameters.
    trainable_params: concatenated [mu, flattened M, epsilon]
    """
    num_params_mu = num_species
    num_params_M = num_species * num_species
    mu = trainable_params[:num_params_mu]
    M_flat = trainable_params[num_params_mu: num_params_mu + num_params_M]
    epsilon = trainable_params[num_params_mu + num_params_M:]
    M = tf.reshape(M_flat, (num_species, num_species))

    # Use the TensorFlow version of the step perturbation
    u_t = step_perturbation_tf(t)
    mu_expanded = tf.expand_dims(mu, axis=0)  # shape (1, num_species)
    My = tf.matmul(y, M, transpose_b=True)    # shape (None, num_species)
    eps_expanded = tf.expand_dims(epsilon, axis=0)
    u_expanded = tf.expand_dims(u_t, axis=1)

    growth = mu_expanded + My + eps_expanded * u_expanded
    dy_dt = y * growth
    residuals = []
    for i in range(num_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]
        residuals.append(dyi_dt - dy_dt[:, i:i+1])
    return residuals


def build_feature_transform():
    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return feature_transform

# --- Inference for one simulation file ---


def infer_parameters_from_file(sim_file):
    """
    Given a simulation JSON file (from the first script), this function sets up the PINN problem
    using the observed (noisy) time series, trains a PINN to infer [mu, M, epsilon], and returns
    both the inferred parameters and the true parameters (loaded from the file).
    """
    # Load simulation data
    with open(sim_file, "r") as f:
        sim_data = json.load(f)

    # Extract required fields
    t_span = np.array(sim_data["t_span"], dtype=np.float32)
    t_data = t_span.reshape(-1, 1)
    noisy_solution = np.array(sim_data["noisy_solution"], dtype=np.float32)
    true_mu = np.array(sim_data["mu_true"], dtype=np.float32)
    true_M = np.array(sim_data["M_true"], dtype=np.float32)
    true_epsilon = np.array(sim_data["epsilon_true"], dtype=np.float32)

    num_species = noisy_solution.shape[1]

    # Create geometry and boundary condition observations from the time series
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])
    observes = [dde.PointSetBC(t_data, noisy_solution[:, i:i+1], component=i)
                for i in range(num_species)]

    # Set up initial guess for trainable parameters: [mu_guess, M_guess (flattened), epsilon_guess]
    num_params_mu = num_species
    num_params_M = num_species * num_species
    param_init = np.hstack([
        np.random.uniform(0.5, 1.5, size=num_species),          # guess for mu
        np.random.uniform(-0.2, 0.2, size=(num_params_M,)
                          ),       # guess for M (flattened)
        # guess for epsilon
        np.random.uniform(-0.1, 0.1, size=num_species)
    ]).astype(np.float32)
    trainable_params = tf.Variable(
        param_init, trainable=True, dtype=tf.float32)

    # Define the ODE residual function
    def ode_residual(t, y):
        return glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species)

    data = dde.data.PDE(geom, ode_residual, observes, anchors=t_data)

    # Build the neural network
    net = dde.maps.FNN([1] + [128, 128, 128] + [num_species],
                       activation="swish",
                       kernel_initializer="Glorot normal")
    net.apply_feature_transform(build_feature_transform())

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss="MSE")
    # Train for a given number of epochs; adjust epochs as needed.
    losshistory, train_state = model.train(epochs=20000, display_every=1000)

    # Extract inferred parameters
    inferred = model.sess.run(trainable_params)
    inferred_mu = inferred[:num_params_mu]
    inferred_M = inferred[num_params_mu: num_params_mu +
                          num_params_M].reshape((num_species, num_species))
    inferred_epsilon = inferred[num_params_mu + num_params_M:]

    # (Optional) Plot predicted vs. observed for visual check
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

    # Return a dictionary of true and inferred parameters
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

# --- Main loop: Process all simulation files and save results ---


def main():
    sim_dir = "simulations"
    out_dir = "inference_results"
    os.makedirs(out_dir, exist_ok=True)

    # Get list of simulation JSON files
    sim_files = [os.path.join(sim_dir, f)
                 for f in os.listdir(sim_dir) if f.endswith(".json")]

    all_results = []
    for sim_file in sim_files:
        print(f"Processing {sim_file} ...")
        result = infer_parameters_from_file(sim_file)
        all_results.append(result)
        # Save individual result
        out_file = os.path.join(out_dir, sim_file.split(
            os.sep)[-1].replace(".json", "_inferred.json"))
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved inferred parameters to {out_file}")

    # Save all results in a combined file
    combined_file = os.path.join(out_dir, "all_inferred_parameters.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved all inferred parameters to {combined_file}")


if __name__ == "__main__":
    main()

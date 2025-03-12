import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from itertools import product

# Scaling factors for the parameters
s_M = 10.0    # Interaction matrix scaling factor
s_eps = 5.0   # Perturbation scaling factor


def step_perturbation_tf(t):
    # Returns 1.0 when t >= 5, else 0.
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species, s_M, s_eps, lambda_pde):
    """
    Compute the PDE residuals for the gLV model with unknown perturbation.
    """
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
        residuals.append(lambda_pde * (dyi_dt - dy_dt[:, i:i+1]))
    return residuals


def build_feature_transform():
    def feature_transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return feature_transform


def load_simulation(sim_file):
    with open(sim_file, "r") as f:
        sim_data = json.load(f)
    t_span = np.array(sim_data["t_span"], dtype=np.float32)
    t_data = t_span.reshape(-1, 1)
    noisy_solution = np.array(sim_data["noisy_solution"], dtype=np.float32)
    return t_span, t_data, noisy_solution


def build_model(t_data, noisy_solution, num_species, trainable_params, lambda_pde, reg_coeff, lr):
    geom = dde.geometry.TimeDomain(t_data[0, 0], t_data[-1, 0])
    observes = [dde.PointSetBC(t_data, noisy_solution[:, i:i+1], component=i)
                for i in range(num_species)]

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
    return model


def tune_simulation(sim_file, tuning_iterations=5000):
    """
    For a single simulation file, run a grid search over hyperparameters,
    and return the best hyperparameters (that minimize the MSE) for that simulation.
    """
    t_span, t_data, noisy_solution = load_simulation(sim_file)
    num_species = noisy_solution.shape[1]

    # Define candidate hyperparameters.
    lambda_pde_values = [1.0, 5.0, 10.0]
    reg_coeff_values = [0.001, 0.01, 0.1]
    learning_rates = [1e-3, 5e-4]

    best_mse = np.inf
    best_config = None

    # Iterate over all combinations for this simulation
    for lambda_pde_val, reg_coeff, lr in product(lambda_pde_values, reg_coeff_values, learning_rates):
        # Initial guess for the parameters:
        num_params_mu = num_species
        num_params_M = num_species * num_species
        mu_guess = np.random.uniform(0.8, 1.6, size=num_species)
        M_guess = np.random.uniform(-0.03, 0.06, size=(num_params_M,))
        for i in range(num_species):
            M_guess[i * num_species +
                    i] = np.random.uniform(-0.15, -0.05) * s_M
        eps_guess = np.random.uniform(-0.15, 0.25, size=num_species) * s_eps
        param_init = np.hstack(
            [mu_guess, M_guess, eps_guess]).astype(np.float32)
        trainable_params = tf.Variable(
            param_init, trainable=True, dtype=tf.float32)

        model = build_model(t_data, noisy_solution, num_species,
                            trainable_params, lambda_pde_val, reg_coeff, lr)
        model.train(iterations=tuning_iterations,
                    display_every=tuning_iterations//5)
        y_pred = model.predict(t_data)
        mse = np.mean((noisy_solution - y_pred)**2)

        if mse < best_mse:
            best_mse = mse
            best_config = {"lambda_pde": lambda_pde_val,
                           "reg_coeff": reg_coeff, "lr": lr}

    print(
        f"For simulation {os.path.basename(sim_file)} best config: {best_config} with MSE: {best_mse:.4e}")
    return best_config


def tune_all_simulations(sim_dir="simulations", tuning_iterations=5000):
    """
    Tune hyperparameters for each simulation individually, then average the best hyperparameters.
    """
    sim_files = [os.path.join(sim_dir, f)
                 for f in os.listdir(sim_dir) if f.endswith(".json")]
    if not sim_files:
        raise ValueError(
            "No simulation files found in the simulations folder.")

    best_configs = []
    for sim_file in sim_files:
        best_config = tune_simulation(
            sim_file, tuning_iterations=tuning_iterations)
        best_configs.append(best_config)

    # Compute the average for each hyperparameter.
    avg_lambda_pde = np.mean([cfg["lambda_pde"] for cfg in best_configs])
    avg_reg_coeff = np.mean([cfg["reg_coeff"] for cfg in best_configs])
    avg_lr = np.mean([cfg["lr"] for cfg in best_configs])

    averaged_config = {"lambda_pde": float(avg_lambda_pde),
                       "reg_coeff": float(avg_reg_coeff),
                       "lr": float(avg_lr)}
    print("\nAveraged hyperparameters over all simulations:")
    print(averaged_config)
    return averaged_config


def main():
    # Tune hyperparameters for each simulation and compute the average.
    averaged_config = tune_all_simulations(
        sim_dir="simulations", tuning_iterations=5000)
    out_dir = "inference_results"
    os.makedirs(out_dir, exist_ok=True)
    best_config_file = os.path.join(out_dir, "averaged_hyperparameters.json")
    with open(best_config_file, "w") as f:
        json.dump(averaged_config, f, indent=4)
    print(f"Averaged hyperparameters saved to {best_config_file}")


if __name__ == "__main__":
    main()
# The script will tune the hyperparameters for each simulation file in the `simulations` folder
# and save the averaged hyperparameters to a JSON file in the `inference_results` folder.
# The hyperparameters are averaged over all simulations, and the best hyperparameters for each
# simulation are printed to the console.

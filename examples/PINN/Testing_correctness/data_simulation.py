import numpy as np
import os
import json
from scipy.integrate import odeint

# Define the step function used in the ODE model


def step_perturbation_np(t):
    # For example, a step that activates at t>=5
    return 1.0 if t >= 5.0 else 0.0

# Define the true generalized Lotka–Volterra ODE with perturbation


def true_glv_ode_with_perturbation(N, t, mu, M, epsilon):
    # Compute the step perturbation at time t (using NumPy version)
    u_t = step_perturbation_np(t)
    # Growth is given by intrinsic growth rate plus interactions and external perturbation
    growth = mu + np.dot(M, N) + epsilon * u_t
    return N * growth

# Generate synthetic data by integrating the ODE and adding noise


def generate_data(num_species, mu, M, epsilon, t_span, init_species, noise_std=0.1):
    # Wrap the ODE function for odeint
    def ode_wrapper(N, t):
        return true_glv_ode_with_perturbation(N, t, mu, M, epsilon)

    # Solve the ODE
    sol = odeint(ode_wrapper, init_species, t_span)
    # Add Gaussian noise to simulate measurement error
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return sol, noisy_sol

# Generate a random set of parameters within the specified ranges:
#   mu: Uniform[0.5, 1.5]
#   M: off-diagonals Uniform[-1, 1] and diagonals fixed to -1 (self-limiting)
#   epsilon: Uniform[-0.2, 0.2]


def generate_random_parameters(num_species):
    mu = np.random.uniform(0.5, 1.5, size=num_species)
    M = np.random.uniform(-1, 1, size=(num_species, num_species))
    np.fill_diagonal(M, -1.0)
    epsilon = np.random.uniform(-0.2, 0.2, size=num_species)
    return mu.astype(np.float32), M.astype(np.float32), epsilon.astype(np.float32)


def main():
    # Number of simulations to generate
    num_simulations = 50
    # Number of species in each simulation (e.g., 3 species)
    num_species = 3
    # Define time span for the simulation: 101 points from t=0 to t=10
    t_span = np.linspace(0, 10, 101).astype(np.float32)
    # Initial condition: all species start at abundance 10
    init_species = np.full(num_species, 0.1, dtype=np.float32)
    # Standard deviation of the additive noise
    noise_std = 0.1

    # Create a directory to save simulations
    os.makedirs("simulations", exist_ok=True)

    for i in range(num_simulations):
        # Generate a random parameter set
        mu, M, epsilon = generate_random_parameters(num_species)
        # Run the simulation
        sol, noisy_sol = generate_data(
            num_species, mu, M, epsilon, t_span, init_species, noise_std)

        # Prepare a dictionary with all the simulation data and the true parameters
        sim_data = {
            "mu_true": mu.tolist(),
            "M_true": M.tolist(),
            "epsilon_true": epsilon.tolist(),
            "t_span": t_span.tolist(),
            "init_species": init_species.tolist(),
            "solution": sol.tolist(),          # Noise-free ODE solution
            "noisy_solution": noisy_sol.tolist()  # Observed data with noise
        }

        # Save each simulation as a JSON file (e.g., simulation_001.json, simulation_002.json, ...)
        filename = f"simulations/simulation_{i+1:03d}.json"
        with open(filename, "w") as f:
            json.dump(sim_data, f)
        print(f"Saved simulation {i+1} to {filename}")


if __name__ == "__main__":
    main()

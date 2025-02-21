import numpy as np
import os
import json
from scipy.integrate import odeint


def step_perturbation_np(t):
    # Step function: activates at t>=5
    return 1.0 if t >= 5.0 else 0.0


def true_glv_ode_with_perturbation(N, t, mu, M, epsilon):
    """
    Generalized Lotka–Volterra ODE with an external step perturbation.
    N: species abundances (vector)
    mu: intrinsic growth rates (vector)
    M: interaction matrix (num_species x num_species)
    epsilon: perturbation vector (vector)
    """
    u_t = step_perturbation_np(t)
    growth = mu + np.dot(M, N) + epsilon * u_t
    return N * growth


def generate_data(num_species, mu, M, epsilon, t_span, init_species, noise_std=0.3):
    """
    Integrate the ODE to generate the noise-free solution and add Gaussian noise.
    """
    def ode_wrapper(N, t):
        return true_glv_ode_with_perturbation(N, t, mu, M, epsilon)
    sol = odeint(ode_wrapper, init_species, t_span)
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return sol, noisy_sol


def generate_random_parameters(num_species):
    """
    Generate a random parameter set based on ranges aligned with the original script:

    - Intrinsic growth rates μ: Uniform[0.8, 1.6]
    - Interaction matrix M:
         * Diagonals (self-interaction): Uniform[-0.16, -0.04]
         * Off-diagonals: Uniform[-0.03, 0.06]
    - Perturbation vector ε: Uniform[-0.15, 0.25]
    """
    # Sample μ for each species
    mu = np.random.uniform(0.8, 1.6, size=num_species)

    # Create an interaction matrix M: first sample off-diagonals, then set diagonals separately.
    M = np.random.uniform(-0.03, 0.06, size=(num_species, num_species))
    diag_vals = np.random.uniform(-0.16, -0.04, size=num_species)
    np.fill_diagonal(M, diag_vals)

    # Sample perturbation vector ε
    epsilon = np.random.uniform(-0.15, 0.25, size=num_species)

    return mu.astype(np.float32), M.astype(np.float32), epsilon.astype(np.float32)


def main():
    # Settings
    num_simulations = 50
    num_species = 3
    # Time span: 101 points from t=0 to t=10
    t_span = np.linspace(0, 10, 101).astype(np.float32)
    # Initial condition: all species start at abundance 10
    init_species = np.full(num_species, 10.0, dtype=np.float32)
    noise_std = 0.3  # Standard deviation of measurement noise

    # Create directory to save simulation files
    os.makedirs("simulations", exist_ok=True)

    for i in range(num_simulations):
        # Generate a random parameter set from the defined ranges
        mu, M, epsilon = generate_random_parameters(num_species)

        # Generate simulation data (noise-free and noisy)
        sol, noisy_sol = generate_data(
            num_species, mu, M, epsilon, t_span, init_species, noise_std)

        # Package all simulation data along with the "true" parameter set
        sim_data = {
            "mu_true": mu.tolist(),
            "M_true": M.tolist(),
            "epsilon_true": epsilon.tolist(),
            "t_span": t_span.tolist(),
            "init_species": init_species.tolist(),
            "solution": sol.tolist(),          # Noise-free solution
            "noisy_solution": noisy_sol.tolist()  # Observed data with noise
        }

        # Save each simulation as a JSON file (e.g., simulation_001.json, simulation_002.json, ...)
        filename = f"simulations/simulation_{i+1:03d}.json"
        with open(filename, "w") as f:
            json.dump(sim_data, f, indent=4)
        print(f"Saved simulation {i+1} to {filename}")


if __name__ == "__main__":
    main()

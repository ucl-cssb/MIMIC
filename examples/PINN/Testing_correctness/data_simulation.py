"""
Stable gLV generator with step perturbation (t = 5) and 3 replicates
====================================================================
* Draws parameters from safer ranges
* Rejects any draw whose trajectory explodes ( > 1e3 or NaNs / Infs )
* JSON layout is exactly what original_parameter_infer.py expects
* Files are written to: sim_reps_mimic/sim_###.json
"""

import json
import os
import numpy as np
from scipy.integrate import odeint


# ------------------------------------------------------------------
# 1.  gLV system with a single step perturbation
# ------------------------------------------------------------------
def step_perturbation(t: float) -> float:
    """0 for t < 5, 1 for t >= 5."""
    return 1.0 if t >= 5.0 else 0.0


def glv_ode(N, t, mu, M, epsilon):
    """Generalised Lotka–Volterra with external perturbation ε·u(t)."""
    u_t = step_perturbation(t)
    growth = mu + M @ N + epsilon * u_t
    return N * growth


# ------------------------------------------------------------------
# 2.  parameter sampling
# ------------------------------------------------------------------
def draw_parameters(n_species: int):
    """Return μ, M, ε drawn from *stable* ranges."""
    mu = np.random.uniform(0.8, 1.6, size=n_species)

    # weaker positive couplings, stronger self‑inhibition
    M = np.random.uniform(-0.02, 0.03, size=(n_species, n_species))
    np.fill_diagonal(M,
                     np.random.uniform(-0.30, -0.10, size=n_species))

    # milder external shock
    epsilon = np.random.uniform(-0.15, 0.25, size=n_species)

    return mu.astype(np.float32), M.astype(np.float32), epsilon.astype(np.float32)


def simulate_until_finite(n_species, t_span, init):
    """Keep drawing parameters until trajectory stays finite and < 1e3."""
    while True:
        mu, M, eps = draw_parameters(n_species)
        sol = odeint(glv_ode, init, t_span, args=(mu, M, eps))

        if (np.isfinite(sol).all()) and (sol.max() < 1e3):
            return mu, M, eps, sol


# ------------------------------------------------------------------
# 3.  main data‑generation loop
# ------------------------------------------------------------------
def main():
    # user settings -------------------------------------------------
    n_sims = 50
    n_species = 6
    n_reps = 3
    noise_std = 0.3                    # >0 for Gaussian measurement noise
    t_span = np.linspace(0, 10, 101).astype(np.float32)
    init_species = np.full(n_species, 10.0, dtype=np.float32)  # higher start

    out_dir = "simulation_replicates"
    os.makedirs(out_dir, exist_ok=True)

    # generation ----------------------------------------------------
    for idx in range(1, n_sims + 1):
        mu, M, eps, sol = simulate_until_finite(n_species,
                                                t_span,
                                                init_species)

        # build replicates
        replicates = []
        for _ in range(n_reps):
            noisy = sol + np.random.normal(0, noise_std, sol.shape)
            replicates.append({"noisy_solution": noisy.tolist()})

        # save JSON
        payload = {
            "mu_true":      mu.tolist(),
            "M_true":       M.tolist(),
            "epsilon_true": eps.tolist(),          # flattened
            "t_span":       t_span.tolist(),
            "init_species": init_species.tolist(),
            "replicates":   replicates
        }

        fname = os.path.join(out_dir, f"sim_{idx:03d}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"✔ saved {fname}")

    print("Finished generating data.")


if __name__ == "__main__":
    main()

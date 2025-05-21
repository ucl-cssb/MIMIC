"""
Smooth‑step gLV generator (3 replicates) – stable ranges
→ outputs to simulation_replicates/sim_###.json
"""

import json
import os
import numpy as np
from scipy.integrate import odeint


# ---------- smooth step -------------------------------------------------
def smooth_step(t, k=8.0, t0=5.0):
    """Logistic step of width 4/k  (≈0.5 for k=8)."""
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))


def glv_ode(N, t, mu, M, eps, k):
    u_t = smooth_step(t, k)
    return N * (mu + M @ N + eps * u_t)


# ---------- parameter draw ----------------------------------------------
def draw_params(S):
    mu = np.random.uniform(0.8, 1.6,  S)
    M = np.random.uniform(-0.02, 0.03, (S, S))
    np.fill_diagonal(M, np.random.uniform(-0.30, -0.10, S))
    eps = np.random.uniform(-0.05, 0.05, S)        # milder shock
    return mu.astype(np.float32), M.astype(np.float32), eps.astype(np.float32)


# ---------- simulator ---------------------------------------------------
def simulate_one(S, t_vec, N0, k):
    while True:                                    # reject explosive draws
        mu, M, eps = draw_params(S)
        sol = odeint(glv_ode, N0, t_vec, args=(mu, M, eps, k))
        if np.isfinite(sol).all() and sol.max() < 1e3:
            return mu, M, eps, sol


# ---------- main loop ---------------------------------------------------
def main():
    S, n_sims, n_reps = 6, 50, 3
    noise = 0.00
    t_vec = np.linspace(0, 10, 101, dtype=np.float32)
    N0 = np.full(S, 10.0, dtype=np.float32)
    k_log = 8.0                                   # step steepness

    out = "simulation_replicates"
    os.makedirs(out, exist_ok=True)

    for idx in range(1, n_sims + 1):
        mu, M, eps, clean = simulate_one(S, t_vec, N0, k_log)

        reps = [{"noisy_solution": (clean + np.random.normal(0, noise, clean.shape)).tolist()}
                for _ in range(n_reps)]

        json.dump({
            "mu_true": mu.tolist(),
            "M_true":  M.tolist(),
            "epsilon_true": eps.tolist(),
            "t_span": t_vec.tolist(),
            "init_species": N0.tolist(),
            "replicates": reps
        }, open(f"{out}/sim_{idx:03d}.json", "w"), indent=2)
        print(f"✔ sim_{idx:03d}.json saved")

    print("all simulations generated")


if __name__ == "__main__":
    main()

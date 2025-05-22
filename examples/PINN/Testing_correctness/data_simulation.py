"""
Sharp-step gLV generator (3 replicates) – safer parameter ranges
----------------------------------------------------------------
• Uses an *abrupt* Heaviside u(t)  (0 for t<5, 1 for t>=5)
• Keeps stronger self-inhibition and milder off-diag couplings
• Rejects explosive trajectories (>1e3 or NaN/Inf)
• Writes JSON files to  simulation_replicates/sim_###.json
"""

import json, os, numpy as np
from  scipy.integrate import odeint


# ---------- 0/1 step ----------------------------------------------------
def step(t, t0=5.0):
    return 1.0 if t >= t0 else 0.0


def glv_ode(N, t, mu, M, eps):
    return N * (mu + M @ N + eps * step(t))


# ---------- parameter draw ---------------------------------------------
def draw_params(S):
    mu  = np.random.uniform(0.8, 1.6,  S)
    M   = np.random.uniform(-0.02, 0.03, (S, S))
    np.fill_diagonal(M, np.random.uniform(-0.30, -0.10, S))
    eps = np.random.uniform(-0.10, 0.10, S)         # milder shock
    return mu.astype(np.float32), M.astype(np.float32), eps.astype(np.float32)


# ---------- simulate until stable --------------------------------------
def simulate_one(S, t_vec, N0):
    while True:
        mu, M, eps = draw_params(S)
        sol = odeint(glv_ode, N0, t_vec, args=(mu, M, eps))
        if np.isfinite(sol).all() and sol.max() < 1e3:
            return mu, M, eps, sol


# ---------- main loop ---------------------------------------------------
def main():
    S, n_sims, n_reps = 6, 50, 3
    noise   = 0.30
    t_vec   = np.linspace(0, 10, 101, dtype=np.float32)
    N0      = np.full(S, 10.0, dtype=np.float32)

    out = "simulation_replicates"
    os.makedirs(out, exist_ok=True)

    for idx in range(1, n_sims + 1):
        mu, M, eps, clean = simulate_one(S, t_vec, N0)

        reps = [{"noisy_solution":
                 (clean + np.random.normal(0, noise, clean.shape)).tolist()}
                for _ in range(n_reps)]

        json.dump({
            "mu_true":      mu.tolist(),
            "M_true":       M.tolist(),
            "epsilon_true": eps.tolist(),          # shape (S,)
            "t_span":       t_vec.tolist(),
            "init_species": N0.tolist(),
            "replicates":   reps
        }, open(f"{out}/sim_{idx:03d}.json", "w"), indent=2)
        print(f"saved  sim_{idx:03d}.json")

    print("all simulations generated")


if __name__ == "__main__":
    main()

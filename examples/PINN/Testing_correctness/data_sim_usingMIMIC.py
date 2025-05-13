import numpy as np
import os
import json
from mimic.model_simulate.base_model import BaseModel
from mimic.model_simulate.sim_gLV import sim_gLV


def step_u(t):
    """
    Step perturbation function: returns array of length 1,
    with 0 for t < 5.0 and 1 for t >= 5.0.
    """
    return np.array([1.0 if t >= 5.0 else 0.0])


def main():
    # --- user settings ---
    num_sims = 50
    num_species = 6
    num_reps = 3
    t_span = np.linspace(0, 10, 101)
    init_species = np.full(num_species, 1.0)
    noise_std = 0.3              # set to 0.0 for noiseless
    out_dir = "sim_reps_mimic"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num_sims):
        # 1) instantiate model with one external perturbation
        model = sim_gLV(num_species=num_species, num_perturbations=1)

        # 2) simulate the “true” trajectory with step perturbation u(t)
        sol, _, mu_true, M_true, eps_true = model.simulate(
            t_span, init_species, u=step_u
        )

        # 3) generate noisy replicates
        replicates = []
        for _ in range(num_reps):
            noisy = sol + np.random.normal(0, noise_std, sol.shape)
            replicates.append({"noisy_solution": noisy.tolist()})

        # 4) build JSON payload
        sim_data = {
            "mu_true":      mu_true.tolist(),
            "M_true":       M_true.tolist(),
            "epsilon_true": eps_true.flatten().tolist(),
            "t_span":       t_span.tolist(),
            "init_species": init_species.tolist(),
            "replicates":   replicates,
        }

        # 5) save to file
        fname = os.path.join(out_dir, f"sim_{i+1:03d}.json")
        with open(fname, "w") as f:
            json.dump(sim_data, f, indent=2)
        print(f"✔ Saved {fname}")


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

# Define scaling factors for M and ε.
s_M = 10.0    # Interaction matrix scaling factor
s_eps = 5.0   # Perturbation scaling factor

# Hyperparameters (from your tuning)
lambda_pde = 1.0      # PDE loss weight
reg_coeff = 0.0046    # L2 regularization coefficient
lr = 0.00075          # learning rate


def step_perturbation_tf(t):
    val = tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))
    return tf.reshape(val, [-1])


def glv_pde_with_unknown_perturbation(t, y, trainable_params, num_species):
    """
    PDE residual for gLV with unknown parameters.
    trainable_params = [mu, M_scaled, eps_scaled]
    """
    # unpack
    num_mu = num_species
    num_M = num_species * num_species
    mu = trainable_params[:num_mu]
    M_scaled = trainable_params[num_mu: num_mu + num_M]
    eps_scaled = trainable_params[num_mu + num_M:]
    # un‐scale
    M = tf.reshape(M_scaled, (num_species, num_species)) / s_M
    epsilon = eps_scaled / s_eps

    u_t = step_perturbation_tf(t)
    mu_exp = tf.expand_dims(mu, 0)            # (1, S)
    My = tf.matmul(y, M, transpose_b=True)    # (N, S)
    eps_exp = tf.expand_dims(epsilon, 0)      # (1, S)
    u_exp = tf.expand_dims(u_t, 1)            # (N, 1)

    growth = mu_exp + My + eps_exp * u_exp     # (N, S)
    dy_dt = y * growth                        # (N, S)

    residuals = []
    for i in range(num_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]
        residuals.append(lambda_pde * (dyi_dt - dy_dt[:, i: i + 1]))
    return residuals


def build_feature_transform():
    def transform(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return transform


def infer_parameters_from_file(sim_file):
    """
    Runs one full round of inference on ALL replicates in sim_file,
    then returns true parameters, replicate‐wise inferred parameters,
    and their mean/std.
    """
    # --- 0) reset TF / DeepXDE graph so each call is fresh ---
    tf.keras.backend.clear_session()
    if hasattr(tf, "reset_default_graph"):
        tf.reset_default_graph()
    elif hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf.compat.v1.reset_default_graph()

    # --- 1) load data ---
    with open(sim_file) as f:
        sim = json.load(f)

    t_span = np.array(sim["t_span"], dtype=np.float32).reshape(-1, 1)
    true_mu = np.array(sim["mu_true"], dtype=np.float32)
    true_M = np.array(sim["M_true"], dtype=np.float32)
    true_eps = np.array(sim["epsilon_true"], dtype=np.float32)

    num_species = true_mu.size
    num_reps = len(sim["replicates"])

    # container for per‐replicate inferences
    rep_results = []

    for idx, rep in enumerate(sim["replicates"], 1):
        noisy = np.array(rep["noisy_solution"], dtype=np.float32)

        # geometry & BCs
        geom = dde.geometry.TimeDomain(t_span[0, 0], t_span[-1, 0])
        bcs = [
            dde.PointSetBC(t_span, noisy[:, i: i + 1], component=i)
            for i in range(num_species)
        ]

        # initial parameter guess
        num_mu = num_species
        num_M = num_species * num_species
        mu_g = np.random.uniform(0.8, 1.6, size=num_species)
        M_g = np.random.uniform(-0.03, 0.06, size=(num_M,))
        # stronger self‐interaction guess
        for j in range(num_species):
            M_g[j * num_species + j] = np.random.uniform(-0.15, -0.05) * s_M
        eps_g = np.random.uniform(-0.15, 0.25, size=num_species) * s_eps

        init_params = np.hstack([mu_g, M_g, eps_g]).astype(np.float32)
        trainable = tf.Variable(init_params, dtype=tf.float32, trainable=True)

        # setup PINN
        def pde_res(t, y):
            return glv_pde_with_unknown_perturbation(
                t, y, trainable, num_species
            )

        data = dde.data.PDE(geom, pde_res, bcs, anchors=t_span)
        net = dde.maps.FNN(
            [1, 128, 128, 128, num_species],
            "swish",
            "Glorot normal",
            regularization=["l2", reg_coeff],
        )
        net.apply_feature_transform(build_feature_transform())
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        model.train(iterations=20000, display_every=1000)

        # pull out inferred parameters
        inf = model.sess.run(trainable)
        mu_inf = inf[:num_mu]
        M_inf = (
            inf[num_mu: num_mu + num_M].reshape(num_species, num_species) / s_M)
        eps_inf = inf[num_mu + num_M:] / s_eps

        rep_results.append(
            {
                "replicate": idx,
                "inferred_mu": mu_inf.tolist(),
                "inferred_M": M_inf.tolist(),
                "inferred_epsilon": eps_inf.tolist(),
            }
        )

    # --- 3) aggregate across replicates ---
    mus = np.array([r["inferred_mu"] for r in rep_results])
    Ms = np.array([r["inferred_M"] for r in rep_results])
    epss = np.array([r["inferred_epsilon"] for r in rep_results])

    mu_mean = mus.mean(axis=0)
    mu_std = mus.std(axis=0)
    M_mean = Ms.mean(axis=0)
    M_std = Ms.std(axis=0)
    eps_mean = epss.mean(axis=0)
    eps_std = epss.std(axis=0)

    return {
        "simulation_file": os.path.basename(sim_file),
        "true_mu": true_mu.tolist(),
        "true_M": true_M.tolist(),
        "true_epsilon": true_eps.tolist(),
        "replicates_inferred": rep_results,
        "mu_mean": mu_mean.tolist(),
        "mu_std": mu_std.tolist(),
        "M_mean": M_mean.tolist(),
        "M_std": M_std.tolist(),
        "epsilon_mean": eps_mean.tolist(),
        "epsilon_std": eps_std.tolist(),
    }


def main():
    sim_dir = "simulations_replicates"
    out_dir = "original_inference_results"
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for fn in sorted(os.listdir(sim_dir)):
        if not fn.endswith(".json"):
            continue
        sim_path = os.path.join(sim_dir, fn)
        print(f"→ Inferring on {fn} …")
        res = infer_parameters_from_file(sim_path)
        all_results.append(res)

        out_path = os.path.join(out_dir, fn.replace(".json", "_inferred.json"))
        with open(out_path, "w") as f:
            json.dump(res, f, indent=4)
        print(f"  saved: {out_path}")

    # also dump combined summary
    with open(os.path.join(out_dir, "all_inferred_summary.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    print("All done.")


if __name__ == "__main__":
    main()

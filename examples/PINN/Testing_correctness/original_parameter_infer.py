# -*- coding: utf-8 -*-

"""
PINN parameter-inference for gLV simulations
===========================================

Changes vs. previous version
----------------------------
* uses the *same smooth step* u(t) that the updated simulator employs
* does NOT clamp the solution at t = 0 (first anchor removed) so the
  network can learn a true initial condition
* diagonal entries of M initialised from  -0.30 to -0.10  (same as simulator)
* longer training schedule: 30 000 Adam steps  ->  L-BFGS refinement
* everything kept strictly ASCII
"""

import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1.  global settings
# ------------------------------------------------------------------
S_M = 5.0          # scaling for M
S_EPS = 2.0          # scaling for epsilon

LAMBDA_PDE = 1.0
REG_COEFF = 4.6e-3
LR = 2e-4    # Adam learning rate
STEP_K = 8.0     # steepness of smooth step
STEP_T0 = 5.0     # centre of step


# ------------------------------------------------------------------
# 2.  helper functions
# ------------------------------------------------------------------
def smooth_step_tf(t, k=STEP_K, t0=STEP_T0):
    """Sigmoid step identical to the simulator."""
    return tf.math.sigmoid(k * (t - t0))


def glv_pde(t, y, params, n_species):
    """PDE residual for unknown mu, M, epsilon."""
    n_mu = n_species
    n_M = n_species * n_species

    mu = params[:n_mu]
    M_s = params[n_mu: n_mu + n_M]
    eps_s = params[n_mu + n_M:]

    M = tf.reshape(M_s, (n_species, n_species)) / S_M
    epsilon = eps_s / S_EPS
    u = smooth_step_tf(t)                     # smooth!

    mu_e = tf.expand_dims(mu, 0)
    My = tf.matmul(y, M, transpose_b=True)
    eps_e = tf.expand_dims(epsilon, 0)
    u_e = tf.expand_dims(u, 1)

    rhs = y * (mu_e + My + eps_e * u_e)

    res = []
    for i in range(n_species):
        dyi_dt = tf.gradients(y[:, i], t)[0]
        res.append(LAMBDA_PDE * (dyi_dt - rhs[:, i:i+1]))
    return res


def build_feature_transform():
    def ft(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return ft


def positive_output_transform(_, y):
    """Softplus ensures non-negative abundances."""
    return tf.nn.softplus(y)


def reset_graph():
    tf.keras.backend.clear_session()
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf.compat.v1.reset_default_graph()


# ------------------------------------------------------------------
# 3.  inference for one simulation (all replicates)
# ------------------------------------------------------------------
def infer_from_file(path_json):
    reset_graph()

    with open(path_json, "r") as f:
        sim = json.load(f)

    t_full = np.asarray(sim["t_span"], dtype=np.float32).reshape(-1, 1)
    mu_t = np.asarray(sim["mu_true"],      dtype=np.float32)
    M_t = np.asarray(sim["M_true"],       dtype=np.float32)
    eps_t = np.asarray(sim["epsilon_true"], dtype=np.float32).reshape(-1)

    S = mu_t.size
    n_M = S * S
    results = []

    # geometry for the PDE (whole time window)
    geom = dde.geometry.TimeDomain(t_full[0, 0], t_full[-1, 0])

    for rep_idx, rep in enumerate(sim["replicates"], start=1):
        y_obs = np.asarray(rep["noisy_solution"], dtype=np.float32)

        # --- remove first point so IC is *not* clamped -------------
        anchors = t_full[1:]
        bcs = [dde.PointSetBC(anchors, y_obs[1:, i:i+1], component=i)
               for i in range(S)]

        # --- random initial parameter guess ------------------------
        mu_g = np.random.uniform(0.8, 1.6, S)
        M_g = np.random.uniform(-0.02, 0.03, n_M)
        for i in range(S):
            M_g[i * S + i] = np.random.uniform(-0.30, -0.10) * S_M
        eps_g = np.random.uniform(-0.05, 0.05, S) * S_EPS

        theta0 = np.hstack([mu_g, M_g, eps_g]).astype(np.float32)
        theta = tf.Variable(theta0, trainable=True, dtype=tf.float32)

        def pde_residual(t, y):
            return glv_pde(t, y, theta, S)

        data = dde.data.PDE(geom, pde_residual, bcs, anchors=t_full)

        net = dde.maps.FNN([1, 128, 128, 128, S],
                           activation="swish",
                           kernel_initializer="Glorot normal",
                           regularization=["l2", REG_COEFF])
        net.apply_feature_transform(build_feature_transform())
        net.apply_output_transform(positive_output_transform)

        model = dde.Model(data, net)

        # ---------------- training schedule -----------------------
        model.compile("adam", lr=LR, loss="MSE")
        model.train(iterations=30000, display_every=2000)

        model.compile("L-BFGS")                    # quasi-Newton
        model.train()                              # until converged

        inferred = model.sess.run(theta)

        mu_inf = inferred[:S]
        M_inf = inferred[S:S + n_M].reshape(S, S) / S_M
        eps_inf = inferred[S + n_M:] / S_EPS

        results.append({
            "replicate":        rep_idx,
            "inferred_mu":      mu_inf.tolist(),
            "inferred_M":       M_inf.tolist(),
            "inferred_epsilon": eps_inf.tolist()
        })

        # quick plot
        clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(8, 4))
        y_hat = model.predict(t_full)
        for k in range(S):
            c = clr[k % len(clr)]
            plt.plot(t_full[:, 0], y_obs[:, k], "o", ms=2, color=c)
            plt.plot(t_full[:, 0], y_hat[:, k], "-", lw=1, color=c)
        plt.title(f"{os.path.basename(path_json)}  rep {rep_idx}")
        plt.tight_layout()
        name = os.path.basename(path_json).replace(".json",
                                                   f"_rep{rep_idx}.png")
        plt.savefig(os.path.join("original_inference_results", name))
        plt.close()

    # ----------- aggregate replicates -----------------------------
    mu_arr = np.asarray([r["inferred_mu"] for r in results])
    M_arr = np.asarray([r["inferred_M"] for r in results])
    eps_arr = np.asarray([r["inferred_epsilon"] for r in results])

    return {
        "simulation_file": os.path.basename(path_json),
        "true_mu":      mu_t.tolist(),
        "true_M":       M_t.tolist(),
        "true_epsilon": eps_t.tolist(),
        "replicates_inferred": results,
        "mu_mean":      mu_arr.mean(axis=0).tolist(),
        "mu_std":       mu_arr.std(axis=0).tolist(),
        "M_mean":       M_arr.mean(axis=0).tolist(),
        "M_std":        M_arr.std(axis=0).tolist(),
        "epsilon_mean": eps_arr.mean(axis=0).tolist(),
        "epsilon_std":  eps_arr.std(axis=0).tolist()
    }


# ------------------------------------------------------------------
# 4.  batch runner
# ------------------------------------------------------------------
def main():
    sim_dir = "simulation_replicates"
    out_dir = "original_inference_results"
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    for fname in sorted(os.listdir(sim_dir)):
        if fname.endswith(".json"):
            print("inferring", fname)
            summary = infer_from_file(os.path.join(sim_dir, fname))
            summaries.append(summary)

            out_file = os.path.join(
                out_dir, fname.replace(".json", "_inferred.json"))
            json.dump(summary, open(out_file, "w"), indent=2)

    json.dump(summaries,
              open(os.path.join(out_dir, "all_inferred_summary.json"), "w"),
              indent=2)
    print("all simulations finished")


if __name__ == "__main__":
    main()

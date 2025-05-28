"""
PINN inference for sharp-step gLV data (fixed initial condition)
---------------------------------------------------------------
* Hard Heaviside perturbation  u(t)=H(t-5)
* The exact, known initial concentration (init_species) is enforced
  as a PointSetBC at t = 0  →  network no longer learns the IC
* Strong self-damping prior, milder epsilon prior
* 20 k Adam iterations (no L-BFGS to keep runtime short)
* Three 128-unit hidden layers, softplus output for positivity
"""

import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt


# ---------------- hyper-parameters -------------------------------------
S_M = 5.0          # scaling for interaction matrix
S_EPS = 2.0          # scaling for epsilon vector
LAMBDA = 1.0         # PDE residual weight
REG = 4.6e-3     # L2 regularisation
LR = 2e-4       # Adam learning rate


# ---------------- helper functions -------------------------------------
def step_tf(t):
    """Heaviside step u(t) = 1 if t>=5 else 0."""
    return tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))


def glv_pde(t, y, theta, n_sp):
    """PDE residual for unknown mu, M, eps."""
    n_mu = n_sp
    n_M = n_sp * n_sp
    mu = theta[:n_mu]
    M_s = theta[n_mu: n_mu + n_M]
    eps_s = theta[n_mu + n_M:]

    M = tf.reshape(M_s, (n_sp, n_sp)) / S_M
    eps = eps_s / S_EPS
    u = step_tf(t)

    rhs = y * (tf.expand_dims(mu, 0)
               + tf.matmul(y, M, transpose_b=True)
               + tf.expand_dims(eps, 0) * tf.expand_dims(u, 1))

    return [LAMBDA * (tf.gradients(y[:, i], t)[0] - rhs[:, i:i+1])
            for i in range(n_sp)]


def feature_tf():
    def ft(t):
        return tf.concat([t, tf.sin(t), tf.sin(2.0 * t)], axis=1)
    return ft


def pos_out(_, y):
    """Softplus to keep abundances non-negative."""
    return tf.nn.softplus(y)


def reset_graph():
    tf.keras.backend.clear_session()
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf.compat.v1.reset_default_graph()


# ---------------- inference for one JSON file --------------------------
def infer_file(json_path):
    reset_graph()

    sim = json.load(open(json_path))
    t_all = np.asarray(sim["t_span"], dtype=np.float32).reshape(-1, 1)
    y_rep0 = np.asarray(sim["replicates"][0]["noisy_solution"],
                        dtype=np.float32)
    init_sp = np.asarray(sim["init_species"], dtype=np.float32)   # exact IC

    S = y_rep0.shape[1]
    n_M = S * S

    # ---------- boundary conditions -----------------------------------
    geom = dde.geometry.TimeDomain(t_all[0, 0], t_all[-1, 0])

    # exact initial condition (noise-free)
    t0 = t_all[:1]                                           # shape (1,1)
    bc_ic = [dde.PointSetBC(t0,
                            init_sp[i:i+1].reshape(1, 1),
                            component=i) for i in range(S)]

    # noisy observations for t > 0
    t_obs = t_all[1:]
    bc_obs = [dde.PointSetBC(t_obs,
                             y_rep0[1:, i:i+1],
                             component=i) for i in range(S)]

    bcs = bc_ic + bc_obs

    # ---------- initial guess for parameters --------------------------
    mu0 = np.random.uniform(0.8, 1.6, S)
    M0 = np.random.uniform(-0.02, 0.03, n_M)
    for i in range(S):
        M0[i*S + i] = np.random.uniform(-0.30, -0.10) * S_M
    eps0 = np.random.uniform(-0.10, 0.10, S) * S_EPS
    theta0 = np.hstack([mu0, M0, eps0]).astype(np.float32)
    theta = tf.Variable(theta0, trainable=True, dtype=tf.float32)

    # ---------- DeepXDE data and model --------------------------------
    data = dde.data.PDE(geom,
                        lambda tt, yy: glv_pde(tt, yy, theta, S),
                        bcs,
                        anchors=t_all)

    net = dde.maps.FNN([1, 128, 128, 128, S],
                       activation="swish",
                       kernel_initializer="Glorot normal",
                       regularization=["l2", REG])
    net.apply_feature_transform(feature_tf())
    net.apply_output_transform(pos_out)

    model = dde.Model(data, net)

    model.compile("adam", lr=LR, loss="MSE")
    model.train(iterations=20000, display_every=1000)

    theta_val = model.sess.run(theta)
    mu_hat = theta_val[:S]
    M_hat = theta_val[S:S + n_M].reshape(S, S) / S_M
    eps_hat = theta_val[S + n_M:] / S_EPS

    # ---------- quick diagnostic plot --------------------------------
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_pred = model.predict(t_all)
    plt.figure(figsize=(8, 4))
    for k in range(S):
        c = colours[k % len(colours)]
        plt.plot(t_all[:, 0], y_rep0[:, k], "o", ms=2, color=c)
        plt.plot(t_all[:, 0], y_pred[:, k], "-", lw=1, color=c)
    plt.title(os.path.basename(json_path))
    plt.tight_layout()
    fig_name = os.path.basename(json_path).replace(".json", ".png")
    plt.savefig(os.path.join("original_inference_results", fig_name))
    plt.close()

    return {
        "file": os.path.basename(json_path),
        "mu_hat":  mu_hat.tolist(),
        "M_hat":   M_hat.tolist(),
        "eps_hat": eps_hat.tolist(),
        "mu_true":  sim["mu_true"],
        "M_true":   sim["M_true"],
        "eps_true": sim["epsilon_true"]
    }


# ---------------- batch runner -----------------------------------------
def main():
    sim_dir = "simulation_replicates"
    out_dir = "original_inference_results"
    os.makedirs(out_dir, exist_ok=True)

    summaries = [infer_file(os.path.join(sim_dir, f))
                 for f in sorted(os.listdir(sim_dir))
                 if f.endswith(".json")]

    json.dump(summaries,
              open(os.path.join(out_dir, "all_results.json"), "w"),
              indent=2)
    print("finished all simulations")


if __name__ == "__main__":
    main()

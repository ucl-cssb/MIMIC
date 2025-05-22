"""
PINN inference for sharp-step gLV data
--------------------------------------
* Uses hard Heaviside step  u(t)=H(t-5)
* Boundary conditions include t = 0 (keeps IC fixed)
* Stronger diagonal prior, milder eps prior
* 30k Adam  ➜  L-BFGS
"""

import os
import json
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt


# ---- global hyper-params ----------------------------------------------
S_M = 5.0     # scale for M
S_EPS = 2.0     # scale for eps
LAMBDA = 1.0
REG = 4.6e-3
LR = 2e-4


# ---- step & PDE --------------------------------------------------------
def step_tf(t):
    return tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))


def glv_pde(t, y, theta, S):
    n_mu, n_M = S, S * S
    mu = theta[:n_mu]
    M_s = theta[n_mu: n_mu + n_M]
    eps_s = theta[n_mu + n_M:]

    M = tf.reshape(M_s, (S, S)) / S_M
    eps = eps_s / S_EPS
    u = step_tf(t)

    rhs = y * (tf.expand_dims(mu, 0)
               + tf.matmul(y, M, transpose_b=True)
               + tf.expand_dims(eps, 0) * tf.expand_dims(u, 1))

    res = [LAMBDA * (tf.gradients(y[:, i], t)[0] - rhs[:, i:i+1])
           for i in range(S)]
    return res


def feature_tf():
    def _ft(t):
        return tf.concat([t, tf.sin(t), tf.sin(2*t)], 1)
    return _ft


def pos_out(_, y):      # softplus to enforce positivity
    return tf.nn.softplus(y)


def reset():
    tf.keras.backend.clear_session()
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf.compat.v1.reset_default_graph()


# ---- single file -------------------------------------------------------
def infer_file(path_json):
    reset()

    sim = json.load(open(path_json))
    t = np.asarray(sim["t_span"], dtype=np.float32).reshape(-1, 1)
    y_t = sim["replicates"][0]["noisy_solution"]  # use rep #1 for BCs
    y_t = np.asarray(y_t, dtype=np.float32)

    S = y_t.shape[1]
    n_M = S * S

    # BCs on ALL time points (IC fixed)
    geom = dde.geometry.TimeDomain(t[0, 0], t[-1, 0])
    bcs = [dde.PointSetBC(t, y_t[:, i:i+1], component=i) for i in range(S)]

    # random initial θ = [μ, M_s, eps_s]
    mu0 = np.random.uniform(0.8, 1.6, S)
    M0 = np.random.uniform(-0.02, 0.03, n_M)
    for i in range(S):
        M0[i*S+i] = np.random.uniform(-0.30, -0.10) * S_M
    eps0 = np.random.uniform(-0.10, 0.10, S) * S_EPS
    theta0 = np.hstack([mu0, M0, eps0]).astype(np.float32)
    theta = tf.Variable(theta0, trainable=True, dtype=tf.float32)

    data = dde.data.PDE(geom,
                        lambda tt, yy: glv_pde(tt, yy, theta, S),
                        bcs, anchors=t)

    net = dde.maps.FNN([1, 128, 128, 128, S], "swish", "Glorot normal",
                       regularization=["l2", REG])
    net.apply_feature_transform(feature_tf())
    net.apply_output_transform(pos_out)

    model = dde.Model(data, net)
    # ---------------- shorter training schedule --------------------
    model.compile("adam", lr=LR, loss="MSE")
    model.train(iterations=20000, display_every=1000)
    # ---------------------------------------------------------------
    # (L-BFGS refinement removed to speed things up)

    theta_val = model.sess.run(theta)
    mu_hat = theta_val[:S]
    M_hat = theta_val[S:S+n_M].reshape(S, S) / S_M
    eps_hat = theta_val[S+n_M:] / S_EPS

    # quick plot
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_pred = model.predict(t)
    plt.figure(figsize=(8, 4))
    for k in range(S):
        c = colours[k % len(colours)]
        plt.plot(t[:, 0], y_t[:, k], "o", ms=2, color=c)
        plt.plot(t[:, 0], y_pred[:, k], "-", lw=1, color=c)
    plt.title(os.path.basename(path_json))
    plt.tight_layout()
    png = os.path.basename(path_json).replace(".json", ".png")
    plt.savefig(os.path.join("original_inference_results", png))
    plt.close()

    return {"file": os.path.basename(path_json),
            "mu_hat":  mu_hat.tolist(),
            "M_hat":   M_hat.tolist(),
            "eps_hat": eps_hat.tolist(),
            "mu_true":  sim["mu_true"],
            "M_true":   sim["M_true"],
            "eps_true": sim["epsilon_true"]}


# ---- batch runner ------------------------------------------------------
def main():
    sim_dir = "simulation_replicates"
    out_dir = "original_inference_results"
    os.makedirs(out_dir, exist_ok=True)

    summaries = [infer_file(os.path.join(sim_dir, f))
                 for f in sorted(os.listdir(sim_dir)) if f.endswith(".json")]

    json.dump(summaries,
              open(os.path.join(out_dir, "all_results.json"), "w"), indent=2)
    print("finished all simulations")


if __name__ == "__main__":
    main()

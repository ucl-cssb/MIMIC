"""
PINN inference (sharp step) – relaxed priors, original scaling
--------------------------------------------------------------
* Heaviside u(t)=H(t-5)
* Initial condition fixed at init_species
* Scaling restored to  s_M = 10 , s_eps = 5  (as in perfect run)
* Parameter priors match simulator ranges: diag M [-0.16,-0.04],
  off-diag [-0.03,0.06], epsilon ±0.25
* No soft-plus output, 20 k Adam only
"""
import os, json, numpy as np, deepxde as dde
from   deepxde.backend import tf
import matplotlib.pyplot as plt

# -------- hyper-params --------------------------------------------------
S_M   = 10.0          # back to original
S_EPS = 5.0
LAMBDA= 1.0
REG   = 4.6e-3
LR    = 2e-4

# -------- helper --------------------------------------------------------
def step_tf(t):
    return tf.where(t >= 5.0, tf.ones_like(t), tf.zeros_like(t))

def glv_pde(t, y, theta, S):
    n_mu, n_M = S, S*S
    mu   = theta[:n_mu]
    M_s  = theta[n_mu:n_mu+n_M]
    eps_s= theta[n_mu+n_M:]
    M    = tf.reshape(M_s, (S,S)) / S_M
    eps  = eps_s / S_EPS
    u    = step_tf(t)
    rhs  = y * (tf.expand_dims(mu,0) +
                tf.matmul(y, M, transpose_b=True) +
                tf.expand_dims(eps,0)*tf.expand_dims(u,1))
    return [LAMBDA*(tf.gradients(y[:,i],t)[0] - rhs[:,i:i+1]) for i in range(S)]

def feature_tf():
    def ft(t): return tf.concat([t, tf.sin(t), tf.sin(2*t)], 1)
    return ft

def reset():                       # convenience
    tf.keras.backend.clear_session()
    if hasattr(tf,"compat") and hasattr(tf.compat,"v1"):
        tf.compat.v1.reset_default_graph()

# -------- single file ---------------------------------------------------
def infer(json_path):
    reset()
    sim    = json.load(open(json_path))
    t_all  = np.asarray(sim["t_span"], dtype=np.float32).reshape(-1,1)
    y_obs  = np.asarray(sim["replicates"][0]["noisy_solution"], dtype=np.float32)
    init_s = np.asarray(sim["init_species"], dtype=np.float32)

    S, n_M = y_obs.shape[1], y_obs.shape[1]**2
    geom   = dde.geometry.TimeDomain(t_all[0,0], t_all[-1,0])

    # exact IC BC
    bc_ic  = [dde.PointSetBC(t_all[:1],
                             init_s[i:i+1].reshape(1,1),
                             component=i) for i in range(S)]
    # noisy BCs for t>0
    bc_obs = [dde.PointSetBC(t_all[1:], y_obs[1:,i:i+1], component=i)
              for i in range(S)]
    bcs = bc_ic + bc_obs

    # ----- initial guess with relaxed priors ----------------------------
    mu0  = np.random.uniform(0.8, 1.6, S)
    M0   = np.random.uniform(-0.03, 0.06, n_M)
    for i in range(S):
        M0[i*S+i] = np.random.uniform(-0.16, -0.04) * S_M
    eps0 = np.random.uniform(-0.25, 0.25, S) * S_EPS
    theta0 = np.hstack([mu0, M0, eps0]).astype(np.float32)
    theta  = tf.Variable(theta0, trainable=True, dtype=tf.float32)

    data = dde.data.PDE(geom, lambda t,y: glv_pde(t,y,theta,S),
                        bcs, anchors=t_all)
    net  = dde.maps.FNN([1,128,128,128,S], "swish", "Glorot normal",
                        regularization=["l2", REG])
    net.apply_feature_transform(feature_tf())

    model = dde.Model(data, net)
    model.compile("adam", lr=LR, loss="MSE")
    model.train(iterations=20000, display_every=1000)

    th = model.sess.run(theta)
    mu_hat  = th[:S]
    M_hat   = th[S:S+n_M].reshape(S,S)/S_M
    eps_hat = th[S+n_M:] / S_EPS

    # quick plot --------------------------------------------------------
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_pred= model.predict(t_all)
    plt.figure(figsize=(8,4))
    for k in range(S):
        c = cols[k%len(cols)]
        plt.plot(t_all[:,0], y_obs[:,k], "o", ms=2, color=c)
        plt.plot(t_all[:,0], y_pred[:,k], "-", lw=1, color=c)
    plt.title(os.path.basename(json_path))
    plt.tight_layout()
    plt.savefig(os.path.join("original_inference_results",
                             os.path.basename(json_path).replace(".json",".png")))
    plt.close()

    return dict(file=os.path.basename(json_path),
                mu_hat=mu_hat.tolist(), M_hat=M_hat.tolist(), eps_hat=eps_hat.tolist(),
                mu_true=sim["mu_true"], M_true=sim["M_true"],
                eps_true=sim["epsilon_true"])

# -------- batch runner --------------------------------------------------
def main():
    sim_dir, out_dir = "simulation_replicates", "original_inference_results"
    os.makedirs(out_dir, exist_ok=True)
    res = [infer(os.path.join(sim_dir,f))
           for f in sorted(os.listdir(sim_dir)) if f.endswith(".json")]
    json.dump(res, open(os.path.join(out_dir,"all_results.json"),"w"), indent=2)
    print("done")

if __name__ == "__main__":
    main()

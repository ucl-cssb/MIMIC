# sbinn_glv_infer.py
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from mimic.model_simulate import sim_gLV

# 1. Simulate Data Using sim_gLV


def generate_data():
    num_species = 3
    M = np.zeros((num_species, num_species))
    np.fill_diagonal(M, [-0.05, -0.1, -0.15])
    M[0, 1] = 0.05
    M[1, 0] = -0.02
    mu = np.array([0.8, 1.2, 1.5])
    simulator = sim_gLV(num_species=num_species, M=M, mu=mu)
    init_species = 10 * np.ones(num_species)
    t = np.linspace(0, 10, 100)[:, None]
    yobs, y0, mu, M, _ = simulator.simulate(
        times=t.ravel(), init_species=init_species)
    y_data = yobs + np.random.normal(0, 0.05, yobs.shape)
    return t, y_data, mu, M, y0

# 2. Define the SBINN Model


def sbinn_model(t, y, y0):
    # Define trainable parameters
    mu = tf.Variable([1.0, 1.0, 1.0], dtype=tf.float32, trainable=True)
    M = tf.Variable(np.zeros((3, 3)), dtype=tf.float32, trainable=True)

    # Define the Lotka-Volterra ODE
    def lv_ode(t, y):
        dydt = y * (mu + tf.matmul(y, M))
        return dydt

    # Geometry and data
    geom = dde.geometry.TimeDomain(t[0][0], t[-1][0])
    data = dde.data.TimePDE(
        geom,
        lv_ode,
        [
            dde.icbc.IC(
                geom,
                lambda x, on: y0[0],
                component=0,
                on_initial=lambda x, on: np.isclose(x[0], geom.t0),
            ),
            dde.icbc.IC(
                geom,
                lambda x, on: y0[1],
                component=1,
                on_initial=lambda x, on: np.isclose(x[0], geom.t0),
            ),
            dde.icbc.IC(
                geom,
                lambda x, on: y0[2],
                component=2,
                on_initial=lambda x, on: np.isclose(x[0], geom.t0),
            ),
        ],
        num_domain=10000,
        num_boundary=2,
    )

    # Define the neural network
    net = dde.maps.FNN([1] + [50] * 3 + [3], "tanh", "Glorot uniform")

    # Define the model
    model = dde.Model(data, net)

    # Compile the model
    model.compile("adam", lr=1e-3)

    return model, mu, M

# 3. Train the Model


def train_model(model, epochs=10000):
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 4. Main Function


def main():
    # Generate data
    t, y_data, true_mu, true_M, y0 = generate_data()

    # Define and compile the model
    model, mu, M = sbinn_model(t, y_data, y0)

    # Train the model
    train_model(model)

    # Print inferred parameters
    inferred_mu = mu.numpy()
    inferred_M = M.numpy()
    print("Inferred mu:", inferred_mu)
    print("Inferred M:", inferred_M)


if __name__ == "__main__":
    main()

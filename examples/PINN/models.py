# First code created by Chris Barnes. This was used as an example

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from matplotlib import pyplot as plt


class PinnDataSet(Dataset):

    def __init__(self, T, Y):
        self.T = T
        self.Y = Y
        if len(self.T) != len(self.Y):
            raise Exception("The length of T does not match the length of Y")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _t = self.T[index]
        _y = self.Y[index]

        return _t, _y


def training_loop(loader, n_epochs, optimizer, model, loss_fn, f0):
    train_loss = np.zeros([n_epochs])

    # initialize the model
    model.train()

    for epoch in range(1, n_epochs + 1):

        # do training over batches
        loss_train = 0.0
        count = 0
        for tT, tY in loader:
            loss = loss_fn(model(tT), tY, tT, f0, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        train_loss[epoch-1] = loss_train/len(loader)

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train/len(loader):.4f}")

    return train_loss


class dyn_model(torch.nn.Module):
    def __init__(self, nspecies):
        super(dyn_model, self).__init__()

        self.nspecies = nspecies

        self.layer1 = nn.Linear(1, 50)
        self.non_linear1 = nn.ReLU()
        self.layer2 = nn.Linear(50, 50)
        self.non_linear2 = nn.ReLU()
        self.layer3 = nn.Linear(50, nspecies)

        # make differentiable parameters of the ODE
        # self.mu = nn.Parameter(data=torch.tensor([1.0, 1.0, 1.0], requires_grad=True)).float()

        # Fix mu
        self.mu = torch.tensor([0.8, 1.2, 1.5])

        # real M parameters
        M = np.zeros((nspecies, nspecies))
        np.fill_diagonal(M, [-0.05, -0.1, -0.15])
        M[0, 1] = 0.05
        M[1, 0] = -0.02

        # M = np.zeros((nspecies, nspecies))
        # np.fill_diagonal(M, [-0.1, -0.1, -0.1])

        # Fix M
        # self.M = torch.tensor(M, requires_grad=True).float()

        self.M = nn.Parameter(data=torch.tensor(M, requires_grad=True)).float()

    def forward(self, input):
        output = self.layer1(input)
        output = self.non_linear1(output)
        output = self.layer2(output)
        output = self.non_linear2(output)
        output = self.layer3(output)

        return output


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

# dX/dt according to Lotka-Volterra


def dX_eq(X, t, mu, M):
    inst_growth = mu + (M @ X.T).T
    return torch.mul(X, inst_growth)


class loss_pinn(nn.Module):
    def __init__(self):
        super(loss_pinn, self).__init__()

    def forward(self, x, y, t, f0, model):
        w_ode = 1.0
        w_aux = 1.0

        # data loss
        loss_data = nn.MSELoss()(x, y)

        # model loss
        dX_phys = dX_eq(x, t, model.mu, model.M)
        dX_model = grad(model(t), t)[0]
        loss_ode = torch.mean((dX_phys - dX_model)**2)

        # boundary loss
        t0 = torch.tensor([0]).float()
        # print(t0, f0, model(t0))
        loss_aux = torch.mean((model(t0) - f0)**2)

        total_loss = loss_data.mean() + w_ode * loss_ode + w_aux * loss_aux
        # print(f"Loss: {total_loss.item():.4f}, Data: {loss_data.item():.4f}, ODE: {loss_ode.item():.4f}, Aux: {loss_aux.item():.4f}")

        return total_loss


def compare_params(mu=None, M=None, alpha=None, e=None):
    # each argument is a tuple of true and predicted values (mu, mu_hat)
    if mu is not None:
        print("mu_hat/mu:")
        print(np.array(mu[1]))
        print(np.array(mu[0]))

        fig, ax = plt.subplots()
        ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
                np.array(mu[1]), markerfmt="D", label='mu_hat', linefmt='C0-')
        ax.stem(np.arange(0, len(mu[0]), dtype="int32"),
                np.array(mu[0]), markerfmt="X", label='mu', linefmt='C1-')
        ax.set_xlabel('i')
        ax.set_ylabel('mu[i]')
        ax.legend()
        plt.savefig("plots-compare-mu.pdf")

    if M is not None:
        print("\nM_hat/M:")
        print(np.round(np.array(M[1]), decimals=2))
        print("\n", np.array(M[0]))

        fig, ax = plt.subplots()
        ax.stem(
            np.arange(
                0,
                M[0].shape[0] ** 2),
            np.array(
                M[1]).flatten(),
            markerfmt="D",
            label='M_hat',
            linefmt='C0-')
        ax.stem(
            np.arange(
                0,
                M[0].shape[0] ** 2),
            np.array(
                M[0]).flatten(),
            markerfmt="X",
            label='M',
            linefmt='C1-')
        ax.set_ylabel('M[i,j]')
        ax.legend()
        plt.savefig("plots-compare-M.pdf")

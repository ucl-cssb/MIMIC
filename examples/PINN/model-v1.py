# First code created by Chris Barnes. This was used as an example


import random

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from mimic.utilities.utilities import *
from mimic.model_infer.infer_gLV_bayes import *

from mimic.model_infer import *
from mimic.model_simulate import *

from scipy.integrate import odeint

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import training_loop
from models import dyn_model
from models import loss_pinn
from models import PinnDataSet
from models import compare_params

#
# generate some data
num_species = 3
M = np.zeros((num_species, num_species))
np.fill_diagonal(M, [-0.05, -0.1, -0.15])
M[0, 1] = 0.05
M[1, 0] = -0.02

# construct growth rates matrix
#mu = np.random.lognormal(0.01, 0.5, num_species)
mu = np.array([0.8, 1.2, 1.5])

# instantiate simulator
simulator = sim_gLV(num_species=num_species,
                    M=M,
                    mu=mu)
simulator.print_parameters()

init_species = 10 * np.ones(num_species)
times = np.arange(0, 10, 0.1)
yobs, y0, mu, M, _ = simulator.simulate(times=times, init_species=init_species)

yobs = yobs + np.random.normal(loc=0, scale=0.1, size=yobs.shape)

# Create tensors
tT = torch.from_numpy( times.reshape(-1,1) ).to(torch.float32)
tY = torch.from_numpy( yobs ).to(torch.float32)
tT.requires_grad = True

print(tT.shape)
print(tY.shape)

n_epochs = 2000
learning_rate = 1e-1
batch_size = 32
weight_decay = 0.1

model = dyn_model(num_species)

loader = DataLoader(PinnDataSet(tT, tY), batch_size=batch_size, shuffle=True)
loss_fn = loss_pinn()
f0 = torch.tensor([10.0,10.0,10.0])

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_loss = training_loop(loader, n_epochs = n_epochs, optimizer = optimizer, model = model, loss_fn = loss_fn, f0 = f0)

# create plot of loss
#thin the points for the plot
train_loss = train_loss[::10]
n_epochs = len(train_loss)

fig, ax = plt.subplots()
plt.xlabel("epoch")
plt.ylabel("log10 loss")
plt.plot(range(n_epochs),np.log10(train_loss), label="training")
plt.legend();
plt.savefig("plots-NN-loss.pdf")

# create plot comparing the original data to the model output
model.eval()

yobs_h = model(tT).detach().numpy()
plot_fit_gLV(yobs, yobs_h, times)
plt.savefig("plots-NN-fit.pdf")

# create a plot predicting the next 50 time points
times_pred = np.arange(0, 20, 0.1)
yobs_pred, _, _, _, _ = simulator.simulate(times=times_pred, init_species=init_species)
tT_pred = torch.from_numpy( times_pred.reshape(-1,1) ).to(torch.float32)
yobs_pred_h = model(tT_pred).detach().numpy()
plot_fit_gLV(yobs_pred, yobs_pred_h, times_pred)
plt.savefig("plots-NN-pred.pdf")

# Look at the parameters
mu_h = model.mu.detach().numpy()
M_h = model.M.detach().numpy()

# # convert weights to matrix
# mu_h = model.mu.detach().numpy()

# predictor = sim_gLV(num_species=num_species,
#                     M=M_h,
#                     mu=mu_h)

# yobs_h, _, _, _, _ = predictor.simulate(times=times, init_species=init_species)

# # plot results
# plot_fit_gLV(yobs, yobs_h, times)
# plt.savefig("plots-NN-fit.pdf")

compare_params(mu=(mu, mu_h), M=(M, M_h))
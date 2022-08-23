
from casadi import *
def xdot(x, theta, u):
    n_species = 3
    x = x[0:n_species]
    M = theta[0:n_species**2].reshape((n_species, n_species)).T # transpose as casadi reshape is inverse compared to numpy



    gr = theta[n_species**2:n_species**2 + n_species]



    dx = gr*x + x*M@x + u


    return dx

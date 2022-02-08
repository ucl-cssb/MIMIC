import numpy as np
from scipy.integrate import odeint


def gLV(y, t, nsp, mu, M):
    dN = np.multiply(mu, y) + np.multiply(y, M @ y)
    return dN

# include a step perturbation after time tp for one hour
def gLVp(y, t, nsp, mu, M, tp, epsilon):
    if (t >= tp) and (t < (tp+1)):
        dN = np.multiply(mu, y) + np.multiply(y, M @ y) + np.multiply(y, epsilon)
    else:
        dN = np.multiply(mu, y) + np.multiply(y, M @ y) 
    
    return dN

def gMLV(sy, t, nsp, mu, M, beta):
    y = sy[0:nsp]
    s = sy[nsp:]
    dN = np.multiply(mu, y) + np.multiply(y, M @ y)
    
    # this is simple production
    dS = beta @ y
    
    # this is growth linked production: model need reconsidering
    # dS = beta @ mp.multiply( dN, y )
    
    return np.hstack((dN, dS))


def sim_gLV_2(times, y0):
    # do two species
    nsp = 2
    # y0=[10,10]
    mu = [1, 2] 
    M = [[-0.005, 0.01], [0, -0.01]]

    yobs = odeint(gLV, y0, times, args=(nsp, mu, M))
    return yobs, y0, mu, M


def sim_gLV_5(times, y0, mu=np.array([])):
    # do five species
    # set up interaction matrix for five species
    M = np.zeros((5, 5))
    np.fill_diagonal(M, [-0.05, -0.1, -0.15, -0.01, -0.2])
    M[0, 2] = -0.025
    M[1, 3] = 0.05
    M[4, 0] = 0.02
    
    nsp = 5
    if mu.size == 0:
        mu = np.random.lognormal(0.01, 0.5, nsp)
    
    if 0:
        print("mu:")
        print(mu)
        print("M:")
        print(M)

    yobs = odeint(gLV, y0, times, args=(nsp, mu, M))
    return yobs, y0, mu, M

# This assumes a step function
def sim_gLV_5_pert(times, y0, mu=np.array([])):
    # do five species
    # set up interaction matrix for five species
    M = np.zeros((5, 5))
    np.fill_diagonal(M, [-0.05, -0.1, -0.15, -0.01, -0.2])
    M[0, 2] = -0.025
    M[1, 3] = 0.05
    M[4, 0] = 0.02
    
    nsp = 5
    if mu.size == 0:
        mu = np.random.lognormal(0.01, 0.5, nsp)
    
    if 0:
        print("mu:")
        print(mu)
        print("M:")
        print(M)
        
    tp = 2
    epsilon = np.array([0, -1, 0 , -1, 0])
        
    yobs = odeint(gLVp, y0, times, args=(nsp, mu, M, tp, epsilon))
    return yobs, y0, mu, M

def sim_gMLV_6by5(times, y0, mu=np.array([])):
    # do five species
    # set up interaction matrix for five species (note this is sparser matrix than in gLV case)
    M = np.zeros((5, 5))
    np.fill_diagonal(M, [-0.05, -0.1, -0.15, -0.01, -0.2])
    
    # M[0,2] = -0.025
    M[1, 3] = 0.05
    # M[4,0] = 0.02
    
    nsp = 5
    if mu.size == 0:
        mu = np.random.lognormal(0.01, 0.5, nsp)
    
    nm = 6
    alpha = np.zeros([nm, nsp])
    alpha[1, 4] = 1
    alpha[4, 2] = -0.5
    sy0 = np.hstack((y0, 10*np.ones(nm)))
   
    if 0:
        print("mu:")
        print(mu)
        print("M:")
        print(M)
        print("alpha:")
        print(alpha)

    syobs = odeint(gMLV, sy0, times, args=(nsp, mu, M, alpha))
    yobs = syobs[:, 0:nsp]
    sobs = syobs[:, nsp:]
        
    return yobs, sobs, sy0, mu, M, alpha

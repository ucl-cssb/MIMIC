import numpy as np
import sys

from gMLV import *


class CompositionalLotkaVolterra:
    """
    Inference for compositional Lotka-Volterra.

    .. math::
        \frac{d}{dt} log(\frac{\pi_i(t)}{\pi_D(t)}) = g_i + \sum^D_{j=1} (A_{ij} \pi_j(t)) + \sum^P_{p=1} (B_{ip} u_p(t)

    """

    def __init__(self, P=None, T=None, U=None, denom=None, pseudo_count=1e-3):
        """
        Parameters
        ----------
        P: A list of T_x by D dimensional numpy arrays of
                estimated relative abundances.
        T: A list of T_x by 1 dimensional numpy arrays giving
                the times of each observation x.
        U: An optional list of T_x by P numpy arrays of external
                perturbations for each x.
        denom : integer id for taxa in denominator of log ratio

        """

        self.P = P
        self.T = T

        if P is not None and denom is None:
            self.denom = choose_denom(P)
            self.X = construct_alr(P, self.denom, pseudo_count)
        elif P is not None and denom is not None:
            self.denom = denom
            self.X = construct_alr(P, denom, pseudo_count)
        else:
            self.X = None

        if U is None and self.X is not None:
            self.U = [np.zeros((x.shape[0], 1)) for x in self.X]
            self.no_effects = True
        else:
            self.U = U
            self.no_effects = False

        # Parameter estimates
        self.A = None
        self.g = None
        self.B = None
        self.Q_inv = np.eye(self.P[0].shape[1] - 1) if P is not None else None

        # Regularization parameters
        self.alpha = None
        self.r_A = None
        self.r_g = None
        self.r_B = None

    def get_regularizers(self):
        return self.alpha, self.r_A, self.r_g, self.r_B

    def set_regularizers(self, alpha, r_A, r_g, r_B):
        self.alpha = alpha
        self.r_A = r_A
        self.r_g = r_g
        self.r_B = r_B

    def train(self, verbose=False, folds=10):
        """
        Estimate regularization parameters and CLV model parameters.
        """
        if self.alpha is None or self.r_A is None or self.r_g is None or self.r_B is None:
            if verbose:
                print("Estimating regularizers...")
            self.alpha, self.r_A, self.r_g, self.r_B = estimate_elastic_net_regularizers_cv(self.X, self.P, self.U,
                                                                                            self.T, self.denom,
                                                                                            folds=folds,
                                                                                            no_effects=self.no_effects,
                                                                                            verbose=verbose)

        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = elastic_net_clv(self.X, self.P, self.U, self.T, self.Q_inv, self.alpha, self.r_A,
                                                 self.r_g, self.r_B, verbose=verbose)

        if verbose:
            print()

    def predict(self, p0, times, u=None):
        """Predict relative abundances from initial conditions.

        Parameters
        ----------
            p0     : the initial observation, a D-dim numpy array
            times  : a T by 1 numpy array of sample times
            u      : a T by P numpy array of external perturbations

        Returns
        -------
            y_pred : a T by D numpy array of predicted relative
                     abundances. Since we cannot predict initial
                     conditions, the first entry is set to the array
                     of -1.
        """
        if u is None:
            u = np.zeros((times.shape[0], 1))

        if p0.ndim == 1:
            p0 = p0.reshape((1, p0.size))

        X = construct_alr([p0], self.denom)
        x = X[0]

        # QUESTION: Is this supposed to be a auto-refferenced function? Or is this predict() method supposed to be from another script?
        return predict(x, p0, u, times, self.A, self.g, self.B, self.denom)

    def get_params(self):
        A = np.copy(self.A)
        g = np.copy(self.g)
        B = np.copy(self.B)
        return A, g, B


def choose_denom(P):
    """Pick a denominator for additive log-ratio transformation.
    """
    np.seterr(divide="ignore", invalid="ignore")
    log_change = None
    for p in P:  # for each subject
        s = p.sum(axis=1, keepdims=True)  # sum each taxon across time
        s[s == 0] = 1
        # calculate the log change between timepoints
        deltas = np.log((p/s)[1:]) - np.log((p/s)[:-1])
        if log_change is None:
            log_change = deltas
        else:
            log_change = np.vstack((log_change, deltas))
    np.seterr(divide="warn", invalid="warn")
    # pick taxon with smallest variance in log proportion
    min_idx = -1
    min_var = np.inf
    n_taxa = log_change.shape[1]
    for i in range(n_taxa):
        if not np.all(np.isfinite(log_change[:, i])):
            continue
        var = np.var(log_change[:, i])
        if var < min_var:
            min_idx = i
            min_var = var

    if min_idx == -1:
        print("Error: no valid denominator found", file=sys.stderr)
        exit(1)

    return min_idx


def construct_alr(P, denom, pseudo_count=1e-3):
    """Compute the additive log ratio transformation with a given
    choice of denominator. Assumes zeros have been replaced with
    nonzero values.
    """
    ALR = []
    n_taxa = P[0].shape[1]
    numer = np.array([i for i in range(n_taxa) if i != denom])
    for p in P:
        p = np.copy(p)
        p = (p + pseudo_count) / (p + pseudo_count).sum(axis=1, keepdims=True)
        p /= p.sum(axis=1, keepdims=True)
        alr = (np.log(p[:, numer]).T - np.log(p[:, denom])).T
        ALR.append(alr)
    return ALR


def estimate_elastic_net_regularizers_cv(X, P, U, T, denom, folds, no_effects=False, verbose=False):
    if len(X) == 1:
        print("Error: cannot estimate regularization parameters from single sample", file=sys.stderr)
        exit(1)
    elif len(X) < folds:
        folds = len(X)

    rs = [0.1, 0.5, 0.7, 0.9, 1]
    alphas = [0.1, 1, 10]

    alpha_rA_rg_rB = []
    for alpha in alphas:
        for r_A in rs:
            for r_g in rs:
                if no_effects:
                    alpha_rA_rg_rB.append((alpha, r_A, r_g, 0))
                else:
                    for r_B in rs:
                        alpha_rA_rg_rB.append((alpha, r_A, r_g, r_B))

    np.set_printoptions(suppress=True)
    best_r = 0
    best_sqr_err = np.inf
    for i, (alpha, r_A, r_g, r_B) in enumerate(alpha_rA_rg_rB):
        # print("\tTesting regularization parameter set", i+1, "of", len(alpha_rA_rg_rB), file=sys.stderr)
        sqr_err = 0
        for fold in range(folds):
            train_X = []
            train_P = []
            train_U = []
            train_T = []

            test_X = []
            test_P = []
            test_U = []
            test_T = []
            for i in range(len(X)):
                if i % folds == fold:
                    test_X.append(X[i])
                    test_P.append(P[i])
                    test_U.append(U[i])
                    test_T.append(T[i])

                else:
                    train_X.append(X[i])
                    train_P.append(P[i])
                    train_U.append(U[i])
                    train_T.append(T[i])

            Q_inv = np.eye(train_X[0].shape[1])
            A, g, B = elastic_net_clv(
                train_X, train_P, train_U, train_T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3)
            # FIXME: #28 where is this defined?
            sqr_err += compute_prediction_error(test_X,
                                                test_P, test_U, test_T, A, g, B, denom)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (alpha, r_A, r_g, r_B), "sqr error", sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def elastic_net_clv(X, P, U, T, Q_inv, alpha, r_A, r_g, r_B, tol=1e-3, verbose=False, max_iter=10000):
    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:, :yDim]
        g = AgB[:, yDim:(yDim+1)]
        B = AgB[:, (yDim+1):]

        grad[:, :yDim] += -2*alpha*(1-r_A)*A
        grad[:, yDim:(yDim+1)] += -2*alpha*(1-r_g)*g
        grad[:, (yDim+1):] += -2*alpha*(1-r_B)*B
        return -grad

    def generalized_gradient(AgB, grad, step):
        nxt_AgB = prv_AgB - step*grad

        # threshold A
        A_prox = nxt_AgB[:, :yDim]
        A_prox[A_prox < -step*alpha*r_A] += step*alpha*r_A
        A_prox[A_prox > step*alpha*r_A] -= step*alpha*r_A
        A_prox[np.logical_and(A_prox >= -step*alpha*r_A,
                              A_prox <= step*alpha*r_A)] = 0

        # threshold g
        g_prox = nxt_AgB[:, yDim:(yDim+1)]
        g_prox[g_prox < -step*alpha*r_g] += step*alpha*r_g
        g_prox[g_prox > step*alpha*r_g] -= step*alpha*r_g
        g_prox[np.logical_and(g_prox >= -step*alpha*r_g,
                              g_prox <= step*alpha*r_g)] = 0

        # threshold B
        B_prox = nxt_AgB[:, (yDim+1):]
        B_prox[B_prox < -step*alpha*r_B] += step*alpha*r_B
        B_prox[B_prox > step*alpha*r_B] -= step*alpha*r_B
        B_prox[np.logical_and(B_prox >= -step*alpha*r_B,
                              B_prox <= step*alpha*r_B)] = 0

        AgB_proximal = np.zeros(AgB.shape)
        AgB_proximal[:, :yDim] = A_prox
        AgB_proximal[:, yDim:(yDim+1)] = g_prox
        AgB_proximal[:, (yDim+1):] = B_prox

        return (AgB - AgB_proximal)/step

    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj

    def stack_observations(X, P, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, p, u, times in zip(X, P, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                pt0 = p[t-1]
                gt0 = np.ones(1)
                ut0 = u[t-1]
                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))

        return x_stacked, pgu_stacked

    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]

    AgB = np.zeros((xDim, yDim + 1 + uDim))
    # FIXME: #29 where is this defined?
    A, g, B = ridge_regression_clv(X, P, U, T, np.max(
        (alpha*(1-r_A), 0.01)), np.max((alpha*(1-r_g), 0.01)), np.max((alpha*(1-r_B), 0.01)))
    AgB[:, :yDim] = A
    AgB[:, yDim:(yDim+1)] = np.expand_dims(g, axis=1)
    AgB[:, (yDim+1):] = B

    x_stacked, pgu_stacked = stack_observations(X, P, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        prv_AgB = np.copy(AgB)
        prv_obj = obj

        # line search
        step = 0.1
        grad = gradient(prv_AgB, x_stacked, pgu_stacked)
        gen_grad = generalized_gradient(prv_AgB, grad, step)
        nxt_AgB = prv_AgB - step*gen_grad
        obj = objective(nxt_AgB, x_stacked, pgu_stacked)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step*0.5*np.square(gen_grad).sum():
            step /= 2
            gen_grad = generalized_gradient(prv_AgB, grad, step)
            nxt_AgB = prv_AgB - step*gen_grad
            obj = objective(nxt_AgB, x_stacked, pgu_stacked)

        A = nxt_AgB[:, :yDim]
        g = nxt_AgB[:, yDim:(yDim+1)]
        B = nxt_AgB[:, (yDim+1):]
        AgB[:, :yDim] = A
        AgB[:, yDim:(yDim+1)] = g
        AgB[:, (yDim+1):] = B

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(
                max_iter), file=sys.stderr)
            break

    A = AgB[:, :yDim]
    g = AgB[:, yDim:(yDim+1)].flatten()
    B = AgB[:, (yDim+1):]

    return A, g, B

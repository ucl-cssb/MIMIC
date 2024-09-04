from . import *
from scipy.stats import linregress
from scipy.special import logsumexp
from scipy.integrate import RK45, solve_ivp
import itertools
import sys
from typing import Tuple, List, Union, Any, Optional
import numpy as np
from numpy.typing import NDArray


class CompositionalLotkaVolterra:
    """
    Inference for compositional Lotka-Volterra.

    .. math::
        \frac{d}{dt} log(\frac{\\pi_i(t)}{\\pi_D(t)}) = g_i + \\sum^D_{j=1} (A_{ij} \\pi_j(t)) + \\sum^P_{p=1} (B_{ip} u_p(t)

    """

    def __init__(self, P=None, T=None, U=None, denom=None, pseudo_count=1e-3):
        """
        Parameters
        ----------
        P : A list of T_x by D dimensional numpy arrays of estimated relative abundances.
        T : A list of T_x by 1 dimensional numpy arrays giving the times of each observation x.
        U : An optional list of T_x by P numpy arrays of external perturbations for each x.
        denom : integer, id for taxa in denominator of log ratio
        pseudo_count : float, default=1e-3, a small number to replace zero counts

        """

        self.P = P
        self.T = T

        if P is not None and denom is None:
            self.denom = choose_denom(P)
            self.X: Optional[List[np.ndarray]] = construct_alr(
                P, self.denom, pseudo_count)
        elif P is not None:
            self.denom = denom
            self.X = construct_alr(P, denom, pseudo_count)
        else:
            self.X = None

        if U is None and self.X is not None:
            self.U = [np.zeros((x.shape[0], 1)) for x in self.X]
            self.no_effects = True
        else:
            self.U = U
            self.no_effects = U is not None and all(np.all(u == 0) for u in U)

        # Parameter estimates
        self.A: Optional[NDArray] = None
        self.g: Optional[NDArray] = None
        self.B: Optional[NDArray] = None
        self.Q_inv = np.eye(self.P[0].shape[1] - 1) if P is not None else None

        # Regularization parameters
        self.alpha: Optional[float] = None
        self.r_A: Optional[float] = None
        self.r_g: Optional[float] = None
        self.r_B: Optional[float] = None

    def get_regularizers(
            self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        return self.alpha, self.r_A, self.r_g, self.r_B

    def set_regularizers(self, alpha, r_A, r_g, r_B):
        self.alpha = alpha
        self.r_A = r_A
        self.r_g = r_g
        self.r_B = r_B

    def train(self, verbose=False, folds=10) -> None:
        """
        Estimate regularization parameters and CLV model parameters.
        """
        if self.alpha is None or self.r_A is None or self.r_g is None or self.r_B is None:
            if verbose:
                print("Estimating regularizers...")
            regularizers = estimate_elastic_net_regularizers_cv(
                self.X, self.P, self.U, self.T, self.denom,
                folds=folds, no_effects=self.no_effects, verbose=verbose
            )

            if regularizers is None:
                if verbose:
                    print(
                        "Failed to estimate regularization parameters. Aborting training.")
                return  # or raise an Exception if that's more appropriate for your use case

            self.alpha, self.r_A, self.r_g, self.r_B = regularizers

        if verbose:
            print("Estimating model parameters...")
        self.A, self.g, self.B = elastic_net_clv(
            self.X, self.P, self.U, self.T, self.Q_inv, self.alpha,
            self.r_A, self.r_g, self.r_B, verbose=verbose
        )

        if verbose:
            print()

    def predict(self, p0, times, u=None) -> np.ndarray:
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

        return predict(x, p0, u, times, self.A, self.g, self.B, self.denom)

    def get_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.A is None or self.g is None or self.B is None:
            print("Error: model has not been trained", file=sys.stderr)
            exit(1)
        A = np.copy(self.A)
        g = np.copy(self.g)
        B = np.copy(self.B)
        return A, g, B


def choose_denom(P) -> int:
    """Pick a denominator for additive log-ratio transformation.
    """
    np.seterr(divide="ignore", invalid="ignore")
    log_change: Optional[NDArray] = None
    for p in P:  # for each subject
        s = p.sum(axis=1, keepdims=True)  # sum each taxon across time
        s[s == 0] = 1
        # calculate the log change between timepoints
        deltas = np.log((p / s)[1:]) - np.log((p / s)[:-1])
        log_change = deltas if log_change is None else np.vstack(
            (log_change, deltas))
    np.seterr(divide="warn", invalid="warn")
    # pick taxon with smallest variance in log proportion
    min_idx = -1
    min_var = np.inf
    if log_change is None:
        print("Error: no valid denominator found", file=sys.stderr)
        exit(1)
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


def construct_alr(P, denom, pseudo_count=1e-3) -> list[np.ndarray]:
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


def estimate_elastic_net_regularizers_cv(X,
                                         P,
                                         U,
                                         T,
                                         denom,
                                         folds,
                                         no_effects=False,
                                         verbose=False) -> Union[Tuple[float,
                                                                       float,
                                                                       float,
                                                                       float],
                                                                 None]:
    if len(X) == 1:
        print(
            "Error: cannot estimate regularization parameters from single sample",
            file=sys.stderr)
        exit(1)
    elif len(X) < folds:
        folds = len(X)

    rs = [0.1, 0.5, 0.7, 0.9, 1.0]
    alphas = [0.1, 1.0, 10.0]

    alpha_rA_rg_rB = []
    for alpha, r_A in itertools.product(alphas, rs):
        for r_g in rs:
            if no_effects:
                alpha_rA_rg_rB.append((alpha, r_A, r_g, 0.0))
            else:
                alpha_rA_rg_rB.extend((alpha, r_A, r_g, r_B) for r_B in rs)
    np.set_printoptions(suppress=True)
    best_r: Optional[Tuple[float, float, float, float]] = None
    best_sqr_err = np.inf
    for i, (alpha, r_A, r_g, r_B) in enumerate(alpha_rA_rg_rB):
        # print("\tTesting regularization parameter set", i+1, "of", len(alpha_rA_rg_rB), file=sys.stderr)
        sqr_err = 0.0
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
            sqr_err += compute_prediction_error(test_X,
                                                test_P, test_U, test_T, A, g, B, denom)

        if sqr_err < best_sqr_err:
            best_r = (alpha, r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (alpha, r_A, r_g, r_B), "sqr error", best_sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def elastic_net_clv(X,
                    P,
                    U,
                    T,
                    Q_inv,
                    alpha,
                    r_A,
                    r_g,
                    r_B,
                    tol=1e-3,
                    verbose=False,
                    max_iter=10000) -> tuple[np.ndarray,
                                             np.ndarray,
                                             np.ndarray]:
    def gradient(AgB, x_stacked, pgu_stacked) -> np.ndarray:
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:, :yDim]
        g = AgB[:, yDim:(yDim + 1)]
        B = AgB[:, (yDim + 1):]

        grad[:, :yDim] += -2 * alpha * (1 - r_A) * A
        grad[:, yDim:(yDim + 1)] += -2 * alpha * (1 - r_g) * g
        grad[:, (yDim + 1):] += -2 * alpha * (1 - r_B) * B
        return -grad

    def generalized_gradient(AgB, grad, step) -> np.ndarray:
        nxt_AgB = prv_AgB - step * grad

        # threshold A
        A_prox = nxt_AgB[:, :yDim]
        A_prox[A_prox < -step * alpha * r_A] += step * alpha * r_A
        A_prox[A_prox > step * alpha * r_A] -= step * alpha * r_A
        A_prox[np.logical_and(A_prox >= -step * alpha * r_A,
                              A_prox <= step * alpha * r_A)] = 0

        # threshold g
        g_prox = nxt_AgB[:, yDim:(yDim + 1)]
        g_prox[g_prox < -step * alpha * r_g] += step * alpha * r_g
        g_prox[g_prox > step * alpha * r_g] -= step * alpha * r_g
        g_prox[np.logical_and(g_prox >= -step * alpha * r_g,
                              g_prox <= step * alpha * r_g)] = 0

        # threshold B
        B_prox = nxt_AgB[:, (yDim + 1):]
        B_prox[B_prox < -step * alpha * r_B] += step * alpha * r_B
        B_prox[B_prox > step * alpha * r_B] -= step * alpha * r_B
        B_prox[np.logical_and(B_prox >= -step * alpha * r_B,
                              B_prox <= step * alpha * r_B)] = 0

        AgB_proximal = np.zeros(AgB.shape)
        AgB_proximal[:, :yDim] = A_prox
        AgB_proximal[:, yDim:(yDim + 1)] = g_prox
        AgB_proximal[:, (yDim + 1):] = B_prox

        return (AgB - AgB_proximal) / step

    def objective(AgB, x_stacked, pgu_stacked) -> float:
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5 * (f.dot(Q_inv) * f).sum()

        return -obj

    def stack_observations(X, P, U, T) -> tuple[Any | None, Any | None]:
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, p, u, times in zip(X, P, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t - 1]
                pt0 = p[t - 1]
                gt0 = np.ones(1)
                ut0 = u[t - 1]
                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t - 1]
                    pgu_stacked = dt * pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t - 1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt * pgu))

        return x_stacked, pgu_stacked

    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]

    AgB = np.zeros((xDim, yDim + 1 + uDim))
    A, g, B = ridge_regression_clv(X, P, U, T, np.max(
        (alpha * (1 - r_A), 0.01)), np.max((alpha * (1 - r_g), 0.01)), np.max((alpha * (1 - r_B), 0.01)))
    AgB[:, :yDim] = A
    AgB[:, yDim:(yDim + 1)] = np.expand_dims(g, axis=1)
    AgB[:, (yDim + 1):] = B

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
        nxt_AgB = prv_AgB - step * gen_grad
        obj = objective(nxt_AgB, x_stacked, pgu_stacked)
        while obj > prv_obj - step * \
                (grad * gen_grad).sum() + step * 0.5 * np.square(gen_grad).sum():
            step /= 2
            gen_grad = generalized_gradient(prv_AgB, grad, step)
            nxt_AgB = prv_AgB - step * gen_grad
            obj = objective(nxt_AgB, x_stacked, pgu_stacked)

        A = nxt_AgB[:, :yDim]
        g = nxt_AgB[:, yDim:(yDim + 1)]
        B = nxt_AgB[:, (yDim + 1):]
        AgB[:, :yDim] = A
        AgB[:, yDim:(yDim + 1)] = g
        AgB[:, (yDim + 1):] = B

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(
                max_iter), file=sys.stderr)
            break

    A = AgB[:, :yDim]
    g = AgB[:, yDim:(yDim + 1)].flatten()
    B = AgB[:, (yDim + 1):]

    return A, g, B


def ridge_regression_clv(X, P, U, T, r_A=0, r_g=0,
                         r_B=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes estimates of A, g, and B using least squares.

    Parameters
    ----------
        X : a list of T x yDim-1 numpy arrays
        U : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation

    Returns
    -------
    """
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB_term1 = np.zeros((xDim, yDim + 1 + uDim))
    AgB_term2 = np.zeros((yDim + 1 + uDim, yDim + 1 + uDim))

    for idx, (xi, pi, ui) in enumerate(zip(X, P, U)):
        for t in range(1, xi.shape[0]):
            pt = pi[t]
            pt0 = pi[t - 1]

            xt = xi[t]
            xt0 = xi[t - 1]

            gt0 = np.ones(1)
            ut0 = ui[t - 1]

            pgu = np.concatenate((pt0, gt0, ut0))

            delT = T[idx][t] - T[idx][t - 1]
            AgB_term1 += np.outer((xt - xt0) / delT, pgu)
            AgB_term2 += np.outer(pgu, pgu)

    reg = np.array(
        (([r_A for _ in range(yDim)] + [r_g]) + [r_B for _ in range(uDim)])
    )
    reg = np.diag(reg)
    AgB = AgB_term1.dot(np.linalg.pinv(AgB_term2 + reg))
    A = AgB[:, :yDim]
    g = AgB[:, yDim:(yDim + 1)].flatten()
    B = AgB[:, (yDim + 1):]

    return A, g, B


def estimate_ridge_regularizers_cv(X,
                                   P,
                                   U,
                                   T,
                                   denom,
                                   folds,
                                   no_effects=False,
                                   verbose=False) -> Union[tuple[float,
                                                                 float,
                                                                 float],
                                                           None]:
    if len(X) == 1:
        print(
            "Error: cannot estimate regularization parameters from single sample",
            file=sys.stderr)
        exit(1)
    elif len(X) < folds:
        folds = len(X)

    rs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    rA_rg_rB = []
    for r_A in rs:
        for r_g in rs:
            if no_effects:
                rA_rg_rB.append((r_A, r_g, 0.0))
            else:
                rA_rg_rB.extend((r_A, r_g, r_B) for r_B in rs)
    np.set_printoptions(suppress=True)
    best_r: Optional[Tuple[float, float, float]] = None
    best_sqr_err = np.inf
    for i, (r_A, r_g, r_B) in enumerate(rA_rg_rB):
        # print("\tTesting regularization parameter set", i+1, "of", len(rA_rg_rB), file=sys.stderr)
        sqr_err = 0.0
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
            A, g, B = ridge_regression_clv(
                train_X, train_P, train_U, train_T, r_A, r_g, r_B)
            sqr_err += compute_prediction_error(test_X,
                                                test_P, test_U, test_T, A, g, B, denom)

        if sqr_err < best_sqr_err:
            best_r = (r_A, r_g, r_B)
            best_sqr_err = sqr_err
            print("\tr", (r_A, r_g, r_B), "sqr error", best_sqr_err)
    np.set_printoptions(suppress=False)
    return best_r


def compute_rel_abun(x, denom) -> np.ndarray:
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    z = np.hstack((x, np.zeros((x.shape[0], 1))))
    p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    for i in range(p.shape[1] - 1, denom, -1):
        tmp = np.copy(p[:, i - 1])
        p[:, i - 1] = np.copy(p[:, i])
        p[:, i] = tmp
    return p


# @timeout(5)
def predict(x, p, u, times, A, g, B, denom) -> np.ndarray:
    """Make predictions from initial conditions
    """
    def grad_fn(A, g, B, u, denom):
        def fn(t, x):
            p = compute_rel_abun(x, denom).flatten()
            return g + A.dot(p) + B.dot(u)
        return fn

    p_pred = np.zeros((times.shape[0], x[0].size + 1))
    pt = p[0]
    xt = x[0]

    for i in range(1, times.shape[0]):
        grad = grad_fn(A, g, B, u[i - 1], denom)
        dt = times[i] - times[i - 1]
        ivp = solve_ivp(grad, (0, 0 + dt), xt, method="RK45")
        xt = ivp.y[:, -1]
        pt = compute_rel_abun(xt, denom).flatten()
        p_pred[i] = pt
    return p_pred


def compute_prediction_error(X, P, U, T, A, g, B, denom_ids) -> float:
    def compute_err(p, p_pred):
        err = 0
        ntaxa = p.shape[1]
        err += np.square(p[1:] - p_pred[1:]).sum()
        return err / ntaxa
    err = 0.0
    for x, p, u, t in zip(X, P, U, T):
        try:
            p_pred = predict(x, p, u, t, A, g, B, denom_ids)
            err += compute_err(p, p_pred)
        except TimeoutError:
            err += np.inf
    return err / len(X)


def estimate_relative_abundances(Y, pseudo_count=1e-3) -> list[np.ndarray]:
    """Adds pseudo counts to avoid zeros and compute relative abundances

    Parameters
    ----------
        Y : a list sequencing counts or observed concentrations
            per sequence
        pseudo_count : pseudo count, specific in relation to
                       the relative proportions of each taxon.

    Returns
    -------
        P : relative abundances with pseudo counts
    """
    P = []
    for y in Y:
        p = np.zeros(y.shape)
        for t in range(y.shape[0]):
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            p[t] = pt
        P.append(p)
    return P

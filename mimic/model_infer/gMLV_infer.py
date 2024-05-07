import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from typing import Any, List, Optional
from scipy.integrate import odeint
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     cross_val_score)


class Ridge1(BaseEstimator, RegressorMixin):
    """Custom ridge regression class"""

    def __init__(self, alphas=None, num_species=3):
        self.coef_: Optional[NDArray[Any]] = None
        if alphas is None:
            alphas = [0.1, 0.1]
        self.alphas = alphas
        self.num_species = num_species

    def fit(self, X, y) -> None:
        # print("calling fit")
        self.coef_ = ridge_fit(X.T, y.T, self.alphas, self.num_species)
        # return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.coef_.T

    def get_params(self, deep=True) -> dict:
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alphas": self.alphas, "num_species": self.num_species}

    # def set_params(self, **parameters):
    #    for parameter, value in parameters.items():
    #        setattr(self, parameter, value)
    #    return self


class Ridge2(BaseEstimator, RegressorMixin):
    """Custom ridge regression class"""

    def __init__(self, alphas=None, num_species=3, num_pert=1):
        self.coef_ = None
        if alphas is None:
            alphas = [0.1, 0.1, 0.1]
        self.alphas = alphas
        self.num_species = num_species
        self.num_pert = num_pert

    def fit(self, X, y) -> None:
        # print("calling fit")
        self.coef_ = ridge_fit_pert(
            X.T, y.T, self.alphas, self.num_species, self.num_pert)
        # return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.coef_.T

    def get_params(self, deep=True) -> dict:
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alphas": self.alphas, "num_species": self.num_species, "num_pert": self.num_pert}


def ridge_fit(X, F, alphas, num_species) -> np.ndarray:
    # To do: redo this with transpose X and Y

    # standard least squares
    # beta = np.dot( np.dot(F,np.transpose(X)), la.inv(np.dot(X,np.transpose(X))))

    # compute ridge estimate
    penalty = np.diagflat(
        np.hstack([np.repeat(alphas[0], num_species), alphas[1]]))

    return (F @ X.T) @ la.inv(X @ X.T + penalty)


def ridge_fit_pert(X, F, alphas, num_species, num_pert) -> np.ndarray:
    # To do: redo this with transposed X and Y

    # standard least squares
    # beta = np.dot( np.dot(F,np.transpose(X)), la.inv(np.dot(X,np.transpose(X))))

    # compute ridge estimate
    penalty = np.diagflat(np.hstack(
        [np.repeat(alphas[0], num_species), alphas[1], np.repeat(alphas[2], num_pert)]))

    return np.dot(F, X.T) @ la.inv(X @ X.T + penalty)


# can use
# import importlib
# import gLV_ML
# importlib.reload(gLV_ML);


def ridge_fit_test(tX, tF, num_species, cRidge=Ridge1) -> None:
    # sourcery skip: extract-duplicate-method
    # NOTE: `cRidge` is not defined in this file, so I temporarily replaced it with `Ridge1`
    print("default ridge")
    model = Ridge(alpha=0.01, fit_intercept=False)
    model.fit(tX, tF)
    print(model.coef_)
    print(model.coef_.shape)
    print(model.predict(tX))

    print("custom ridge")
    # FIXME: #27 is this `cRidge` defined anywhere? or is it `Ridge`?
    model = cRidge(alphas=[0.01, 0.01], num_species=num_species)
    model.fit(tX, tF)
    print(model.coef_)
    print(model.coef_.shape)
    print(model.predict(tX))


def linearize_time_course_16S(yobs, times) -> tuple[np.ndarray, np.ndarray]:
    num_species = yobs.shape[1]
    nt = len(times)

    # F = dlnX/dt
    DlnX = np.diff(np.log(yobs), axis=0)
    Dt = np.tile(np.diff(times), (num_species, 1))
    # print(DlnX)
    # print(Dt)
    F = np.divide(DlnX, np.transpose(Dt))
    # print(F)

    # X matrix: stacked observed counts
    X = np.vstack([np.transpose(yobs), np.ones(nt)])
    # print(X)

    # Get data into correct format for scikit-learn
    tF = F
    # print("tF:",np.shape(tF))

    # remove last column of X and transpose to get design matrix
    tX = np.transpose(X[:, 0:-1])
    # print("tX:",np.shape(tX))

    # plot data in one variable
    # plt.scatter(tX[:,0], tF[:,0]);

    return tX, tF


# here u should be of length timepoints
def linearize_time_course_16S_u(yobs, times, u) -> tuple[np.ndarray, np.ndarray]:
    num_species = yobs.shape[1]
    nt = len(times)

    # F = dlnX/dt
    DlnX = np.diff(np.log(yobs), axis=0)
    Dt = np.tile(np.diff(times), (num_species, 1))
    F = np.divide(DlnX, np.transpose(Dt))

    # X matrix: stacked observed counts
    X = np.vstack([np.transpose(yobs), np.ones(nt), u])

    # remove last column of X and transpose to get design matrix
    tX = np.transpose(X[:, 0:-1])
    # print("tX:",np.shape(tX))

    return tX, F


def linearise_time_course_metabolites(sobs, yobs, times) -> tuple[np.ndarray, np.ndarray]:
    nm = sobs.shape[1]
    ns = yobs.shape[1]

    # S = ds/dt
    DS = np.diff(sobs, axis=0)
    Dt = np.tile(np.diff(times), (nm, 1))
    S = np.divide(DS, np.transpose(Dt))

    # X = dX/dt
    DX = np.diff(yobs, axis=0)
    Dt = np.tile(np.diff(times), (ns, 1))
    X = np.divide(DX, np.transpose(Dt))
    # X = yobs[0:-1, :]
    return X, S


def linearize_time_course(yobs, times) -> tuple[np.ndarray, np.ndarray]:
    return linearize_time_course_16S(yobs, times)


def plot_coeffs(tX, tF) -> None:
    n_alphas = 10
    alphas = np.logspace(-5, 2, n_alphas)
    print(alphas)

    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(tX, tF)
        # print( ridge.coef_.flatten() )
        coefs.append(ridge.coef_.flatten())

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()


def fit_alpha_Ridge1(X, F, num_species, n_a0, n_a1) -> tuple[float, float]:
    # use own ridge model

    a0 = np.logspace(-2, 2, n_a0)  # constraint on Mij matrix elements
    a1 = np.logspace(-6, 0, n_a1)  # constraint on mu

    xv, yv = np.meshgrid(a0, a1, indexing='ij')

    candidate_regressors: List[Any] = []
    for i in range(n_a0):
        candidate_regressors.extend(
            Ridge1(alphas=[xv[i, j], yv[i, j]], num_species=num_species)
            for j in range(n_a1)
        )
    cv = RepeatedKFold(n_splits=10, n_repeats=10)
    cv_results = [-cross_val_score(r, X, F, scoring='neg_root_mean_squared_error', cv=cv)
                  for r in candidate_regressors]

    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)
    inds = np.unravel_index(min_i, (n_a0, n_a1))
    print("minimum found: a0/a1/error:",
          a0[inds[0]], a1[inds[1]], cv_means[min_i])

    # unconstrained to compare
    unc_model = Ridge1(alphas=[0, 0], num_species=num_species)
    cv_results = -cross_val_score(unc_model, X, F,
                                  scoring='neg_root_mean_squared_error', cv=cv)
    print("unconstrained error        :", np.mean(cv_results))

    return a0[inds[0]], a1[inds[1]]


def fit_alpha_Ridge2(X, F, num_species, num_pert, n_a0, n_a1, n_a2) -> tuple[float, float, float]:
    # use own ridge model

    a0 = np.logspace(-6, 3, n_a0)  # constraint on Mij matrix elements
    a1 = np.logspace(-6, 3, n_a1)  # constraint on mu
    a2 = np.logspace(-6, 3, n_a2)  # constraint on epsilon

    xv, yv, zv = np.meshgrid(a0, a1, a2, indexing='ij')

    candidate_regressors: List[Any] = []
    for i in range(n_a0):
        for j in range(n_a1):
            candidate_regressors.extend(
                Ridge2(
                    alphas=[xv[i, j, k], yv[i, j, k], zv[i, j, k]],
                    num_species=num_species,
                    num_pert=num_pert,
                )
                for k in range(n_a2)
            )
    cv = RepeatedKFold(n_splits=10, n_repeats=10)
    cv_results = [-cross_val_score(r, X, F, scoring='neg_root_mean_squared_error', cv=cv)
                  for r in candidate_regressors]

    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)
    inds = np.unravel_index(min_i, (n_a0, n_a1, n_a2))
    print("minimum found: a0/a1/a2/error:",
          a0[inds[0]], a1[inds[1]], a2[inds[2]], cv_means[min_i])

    # unconstrained to compare
    unc_model = Ridge2(
        alphas=[0, 0, 0], num_species=num_species, num_pert=num_pert)
    cv_results = -cross_val_score(unc_model, X, F,
                                  scoring='neg_root_mean_squared_error', cv=cv)
    print("unconstrained error        :", np.mean(cv_results))

    return a0[inds[0]], a1[inds[1]], a2[inds[2]]


def do_final_fit_Ridge1(X, F, num_species, a0, a1) -> tuple[list[float], list[list[float]]]:
    model = Ridge1(alphas=[a0, a1], num_species=num_species)
    model.fit(X, F)
    if model.coef_ is None:
        raise ValueError("Model coefficients are not set.")
    mu_h = [model.coef_[i][-1] for i in range(num_species)]
    M_h = [model.coef_[i][:num_species].tolist() for i in range(num_species)]
    return mu_h, M_h


def do_final_fit_Ridge2(X, F, num_species, num_pert, a0, a1, a2) -> tuple[list[float], list[list[float]], list[list[float]]]:
    model = Ridge2(alphas=[a0, a1, a2],
                   num_species=num_species, num_pert=num_pert)
    model.fit(X, F)
    if model.coef_ is None:
        raise ValueError("Model coefficients are not set.")
    M_h = [model.coef_[i][:num_species].tolist() for i in range(num_species)]
    mu_h = [model.coef_[i][num_species] for i in range(num_species)]
    e_h = [model.coef_[i][(num_species+1):] for i in range(num_species)]

    return mu_h, M_h, e_h


def do_bootstrapping(X, F, num_species, a0, a1, nt, nboots=100) -> None:
    # do some bootstrapping
    model = Ridge1(alphas=[a0, a1], num_species=num_species)

    mus = np.zeros([nboots, num_species])
    mms = np.zeros([nboots, num_species * num_species])
    for i in range(nboots):
        sample_index = np.random.choice(range(nt - 1), nt - 1)

        X_s = X[sample_index, :]
        F_s = F[sample_index, :]

        model.fit(X_s, F_s)
        if model.coef_ is None:
            raise ValueError("Model coefficients are not set.")
        mu_h = [model.coef_[i][-1] for i in range(num_species)]
        M_h = [model.coef_[i][:num_species].tolist()
               for i in range(num_species)]

        mus[i, :] = mu_h
        mms[i, :] = np.array(M_h).flatten()

        # print(np.array(mu_h))
        # print(np.round(np.array(M_h),decimals=2))

    print("examining mu_i")
    mus_max = mus.max(axis=0)
    mus_min = mus.min(axis=0)
    for i in range(num_species):
        star = ""
        if np.abs(mus_min[i] - mus_max[i]) > 1e-4:
            if mus_min[i] > 0 and mus_max[i] > 0:
                star = "*"
            elif mus_min[i] < 0 and mus_max[i] < 0:
                star = "*"

        print(i, np.round(mus_min[i], decimals=3),
              " - ", np.round(mus_max[i], decimals=3), star)

    mms_max = mms.max(axis=0)
    mms_min = mms.min(axis=0)
    print("\nexamining Mij")
    for i in range(num_species * num_species):
        star = ""
        if np.abs(mms_min[i] - mms_max[i]) > 1e-4:
            if mms_min[i] > 0 and mms_max[i] > 0:
                star = "*"
            elif mms_min[i] < 0 and mms_max[i] < 0:
                star = "*"

        print(i + 1, np.unravel_index(i, (num_species, num_species)), np.round(mms_min[i], decimals=3), " - ",
              np.round(mms_max[i], decimals=3), star)


def plot_alpha_lasso(X, S, n_a) -> None:
    candidate_alpha = np.logspace(-1, 2, n_a)
    candidate_regressors = [Lasso(
        alpha=a, fit_intercept=False, max_iter=10000, tol=1e-1) for a in candidate_alpha]

    coefs = [r.fit(X, S).coef_.flatten() for r in candidate_regressors]

    plt.figure()
    ax = plt.gca()
    ax.plot(candidate_alpha, coefs)
    ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.ylim(-1, 1)
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Lasso coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()


def fit_alpha_lasso(X, S, n_a) -> tuple[float, float]:
    candidate_alpha = np.logspace(-1, 2, n_a)
    candidate_regressors = [Lasso(
        alpha=a, fit_intercept=False, max_iter=10000, tol=1e-1) for a in candidate_alpha]

    cv = RepeatedKFold(n_splits=10, n_repeats=10)
    cv_results = [-cross_val_score(r, X, S, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1) for r in
                  candidate_regressors]

    n_est = np.array([len(x) for x in cv_results])
    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)
    cutoff = cv_means[min_i] + cv_se[min_i]
    one_se_rule_i = np.argmax(candidate_alpha * (cv_means < cutoff))

    print("minimum found: a/error:", candidate_alpha[min_i], cv_means[min_i])
    print("min + se rule: a/error:",
          candidate_alpha[one_se_rule_i], cv_means[one_se_rule_i])

    plt.figure()
    plt.plot(candidate_alpha, cv_means)
    plt.fill_between(candidate_alpha, cv_means + 1 *
                     cv_se, cv_means - 1 * cv_se, alpha=.1)
    plt.axhline(cutoff, linestyle='dotted', label='Best + One SE')
    plt.scatter([candidate_alpha[one_se_rule_i]], [cv_means[one_se_rule_i]], marker='o', color='orange',
                label='One SE Rule')
    plt.scatter([candidate_alpha[min_i]], [cv_means[min_i]],
                marker='o', color='blue', label='Minimum rule')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Log alpha')
    plt.ylabel('MSE')
    plt.show()

    return candidate_alpha[min_i], candidate_alpha[one_se_rule_i]


###########################################################
# older function using other more standard methods. Might come back to these at some point


# def fit_alpha_default():
#     # find the optimal penalisation terms
#     # model = Ridge(fit_intercept=False)
#     model = Lasso(fit_intercept=False, max_iter=10000)
#     # model = ElasticNet(fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

#     cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)  # five fold
#     n_alphas = 100
#     grid = {'alpha': np.logspace(-6, 0, n_alphas)}
#     # define search
#     search = GridSearchCV(
#         model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
#     # perform the search
#     results = search.fit(tX, tF)

#     # summarize
#     print('MAE: %.3f' % results.best_score_)
#     print(f'Config: {results.best_params_}')

#     # fit using optimal alpha
#     # model = Ridge(alpha=results.best_params_['alpha'], fit_intercept=False)
#     model = Lasso(alpha=results.best_params_[
#                   'alpha'], fit_intercept=False, max_iter=10000)
#     # model = ElasticNet(alpha=results.best_params_['alpha'], fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

#     # model = ElasticNet(alpha=0.01, fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

#     model.fit(tX, tF)
#     mu_h = [model.coef_[i][-1] for i in range(num_species)]
#     M_h = [model.coef_[i][:num_species].tolist() for i in range(num_species)]

#     modelB = LinearRegression(fit_intercept=False)
#     modelB.fit(tX, tF)
#     mu_l = [modelB.coef_[i][-1] for i in range(num_species)]
#     M_l = [modelB.coef_[i][:num_species].tolist() for i in range(num_species)]

#     print("\ninferred params:")
#     print("mu_hat/mu/mu_l:")
#     print(np.array(mu_h))
#     print(np.array(mu))
#     print(np.array(mu_l))
#     print("\nM_hat/M/M_l:")
#     print(np.round(np.array(M_h), decimals=2))
#     print("\n", np.array(M))
#     print("\n", np.round(np.array(M_l), decimals=2))

#     # plot the fit
#     yobs_pred = odeint(gLV, y0, times, args=(num_species, mu_h, M_h))
#     plt.plot(times, yobs)
#     plt.plot(times, yobs_pred, '--')

#     # plot the params
#     plt.figure()
#     plt.stem(np.arange(0, len(mu), dtype="int32"),
#              np.array(mu_h), markerfmt="D")
#     plt.stem(np.arange(0, len(mu), dtype="int32"), np.array(mu), markerfmt="X")

#     plt.figure()
#     plt.stem(np.arange(0, num_species * num_species),
#              np.array(M_h).flatten(), markerfmt="D")
#     plt.stem(np.arange(0, num_species * num_species),
#              np.array(M).flatten(), markerfmt="X")

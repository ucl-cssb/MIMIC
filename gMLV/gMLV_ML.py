import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as la

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from scipy.integrate import odeint

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin


class cRidge(BaseEstimator, RegressorMixin):
    """Custom ridge regression class"""

    def __init__(self, alphas=[0.1, 0.1], nsp=3):
        self.alphas = alphas
        self.nsp = nsp

    def fit(self, X, y):
        # print("calling fit")
        self.coef_ = ridge_fit(X.T, y.T, self.alphas, self.nsp)
        # return self

    def predict(self, X):
        return X @ self.coef_.T

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alphas": self.alphas, "nsp": self.nsp}

    # def set_params(self, **parameters):
    #    for parameter, value in parameters.items():
    #        setattr(self, parameter, value)
    #    return self


def ridge_fit(X, F, alphas, nsp):
    # To do: redo this with transposed X and Y
    
    # standard least squares
    # beta = np.dot( np.dot(F,np.transpose(X)), la.inv(np.dot(X,np.transpose(X))))
    
    # compute ridge estimate
    penalty = np.diagflat(np.hstack([np.repeat(alphas[0], nsp), alphas[1]]))
    
    beta = (F @ X.T) @ la.inv(X @ X.T + penalty) 
    
    return beta


# can use
# import importlib
# import gLV_ML 
# importlib.reload(gLV_ML);

def ridge_fit_test(X, Y):
    print("default ridge")
    model = Ridge(alpha=0.01, fit_intercept=False)
    model.fit(tX, tF)
    print(model.coef_)
    print(model.coef_.shape)
    print(model.predict(tX))

    print("custom ridge")
    model = cRidge(alphas=[0.01, 0.01], nsp=nsp)
    model.fit(tX, tF)
    print(model.coef_)
    print(model.coef_.shape)
    print(model.predict(tX))


def linearize_time_course_16S(yobs, times):
    nsp = yobs.shape[1]
    nt = len(times)

    # F = dlnX/dt
    DlnX = np.diff(np.log(yobs), axis=0)
    Dt = np.tile(np.diff(times), (nsp, 1))
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


def linearise_time_course_metabolites(sobs, yobs, times):
    nm = sobs.shape[1]
    # print("nm:",nm)
    nt = len(times)
    
    # S = ds/dt
    DS = np.diff(sobs, axis=0)
    Dt = np.tile(np.diff(times), (nm, 1))
    S = np.divide(DS, np.transpose(Dt))
    X = yobs[0:-1, :]
    # print("S:",np.shape(S))
    # print("X:",np.shape(X))
    return X, S


def linearize_time_course(yobs, times):
    return linearize_time_course_16S(yobs, times)


def plot_coeffs():
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


def fit_alpha_cRidge(X, F, nsp, n_a0, n_a1):
    # use own ridge model
    
    a0 = np.logspace(-2, 2, n_a0)    # constraint on Mij matrix elements
    a1 = np.logspace(-6, 0, n_a1)    # constraint on mu

    xv, yv = np.meshgrid(a0, a1, indexing='ij')

    candidate_regressors = []
    for i in range(n_a0):
        for j in range(n_a1):
            # print(i, j, xv[i,j], yv[i,j])
            candidate_regressors.append(cRidge(alphas=[xv[i, j], yv[i, j]], nsp=nsp))
    
    cv = RepeatedKFold(n_splits=10, n_repeats=10)
    cv_results = [-cross_val_score(r, X, F, scoring='neg_root_mean_squared_error', cv=cv) for r in candidate_regressors]

    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)
    inds = np.unravel_index(min_i, (n_a0, n_a1))
    print("minimum found: a0/a1/error:", a0[inds[0]], a1[inds[1]], cv_means[min_i])

    # unconstrained to compare
    unc_model = cRidge(alphas=[0, 0], nsp=nsp)
    cv_results = -cross_val_score(unc_model, X, F, scoring='neg_root_mean_squared_error', cv=cv) 
    print("unconstrained error        :", np.mean(cv_results))
    
    return a0[inds[0]], a1[inds[1]]


def do_final_fit_cRidge(X, F, nsp, a0, a1):
    model = cRidge(alphas=[a0, a1], nsp=nsp)
    model.fit(X, F)
    mu_h = [model.coef_[i][-1] for i in range(0, nsp)]
    M_h = [model.coef_[i][0:nsp].tolist() for i in range(0, nsp)]
    return mu_h, M_h


def do_bootstrapping(X, F, nsp, a0, a1, nt, nboots=100):
    # do some bootstrapping 
    model = cRidge(alphas=[a0, a1], nsp=nsp)

    mus = np.zeros([nboots, nsp])
    mms = np.zeros([nboots, nsp*nsp])
    for i in range(0, nboots):
        sample_index = np.random.choice(range(0, nt-1), nt-1)

        X_s = X[sample_index, :]
        F_s = F[sample_index, :]
    
        model.fit(X_s, F_s)
        mu_h = [model.coef_[i][-1] for i in range(0, nsp)]
        M_h = [model.coef_[i][0:nsp].tolist() for i in range(0, nsp)]

        mus[i, :] = mu_h
        mms[i, :] = np.array(M_h).flatten()
    
        # print(np.array(mu_h))
        # print(np.round(np.array(M_h),decimals=2))

    print("examining mu_i")
    mus_max = mus.max(axis=0)
    mus_min = mus.min(axis=0)
    for i in range(0, nsp):
        star = ""
        if np.abs(mus_min[i] - mus_max[i]) > 1e-4:
            if mus_min[i] > 0 and mus_max[i] > 0:
                star = "*"
            elif mus_min[i] < 0 and mus_max[i] < 0:
                star = "*"
        
        print(i, np.round(mus_min[i], decimals=3), " - ", np.round(mus_max[i], decimals=3), star)
     
    mms_max = mms.max(axis=0)
    mms_min = mms.min(axis=0)
    print("\nexamining Mij")
    for i in range(0, nsp*nsp):
        star = ""
        if np.abs(mms_min[i] - mms_max[i]) > 1e-4:
            if mms_min[i] > 0 and mms_max[i] > 0:
                star = "*"
            elif mms_min[i] < 0 and mms_max[i] < 0:
                star = "*"

        print(i+1, np.unravel_index(i, (nsp, nsp)), np.round(mms_min[i], decimals=3), " - ", np.round(mms_max[i], decimals=3), star)


def plot_alpha_lasso(X, S, n_a):
    candidate_alpha = np.logspace(-1, 2, n_a)
    candidate_regressors = [Lasso(alpha=a, fit_intercept=False, max_iter=10000, tol=1e-1) for a in candidate_alpha]
    
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


def fit_alpha_lasso(X, S, n_a):
    
    candidate_alpha = np.logspace(-1, 2, n_a)
    candidate_regressors = [Lasso(alpha=a, fit_intercept=False, max_iter=10000, tol=1e-1) for a in candidate_alpha]
    
    cv = RepeatedKFold(n_splits=10, n_repeats=10)
    cv_results = [-cross_val_score(r, X, S, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1) for r in candidate_regressors]

    n_est = np.array([len(x) for x in cv_results])
    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)
    cutoff = cv_means[min_i] + cv_se[min_i]
    one_se_rule_i = np.argmax(candidate_alpha * (cv_means < cutoff))

    print("minimum found: a/error:", candidate_alpha[min_i], cv_means[min_i])
    print("min + se rule: a/error:", candidate_alpha[one_se_rule_i], cv_means[one_se_rule_i])

    plt.figure()
    plt.plot(candidate_alpha, cv_means)
    plt.fill_between(candidate_alpha, cv_means + 1*cv_se, cv_means - 1*cv_se, alpha=.1)
    plt.axhline(cutoff, linestyle='dotted', label='Best + One SE')
    plt.scatter([candidate_alpha[one_se_rule_i]], [cv_means[one_se_rule_i]], marker='o', color='orange', label='One SE Rule')
    plt.scatter([candidate_alpha[min_i]], [cv_means[min_i]], marker='o', color='blue', label='Minimum rule')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Log alpha')
    plt.ylabel('MSE')
    plt.show()
    
    return candidate_alpha[min_i], candidate_alpha[one_se_rule_i]
    
###########################################################
# older function using other more standard methods. Might come back to these at some point


def fit_alpha_default():
    # find the optimal penalisation terms
    # model = Ridge(fit_intercept=False)
    model = Lasso(fit_intercept=False, max_iter=10000)
    # model = ElasticNet(fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)  # five fold
    n_alphas = 100
    grid = dict()
    grid['alpha'] = np.logspace(-6, 0, n_alphas)

    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(tX, tF)
    
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)

    # fit using optimal alpha
    # model = Ridge(alpha=results.best_params_['alpha'], fit_intercept=False)
    model = Lasso(alpha=results.best_params_['alpha'], fit_intercept=False, max_iter=10000)
    # model = ElasticNet(alpha=results.best_params_['alpha'], fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

    # model = ElasticNet(alpha=0.01, fit_intercept=False, max_iter=100000, l1_ratio=0.9, tol=1e-2)

    model.fit(tX, tF)
    mu_h = [model.coef_[i][-1] for i in range(0, nsp)]
    M_h = [model.coef_[i][0:nsp].tolist() for i in range(0, nsp)]

    modelB = LinearRegression(fit_intercept=False)
    modelB.fit(tX, tF)
    mu_l = [modelB.coef_[i][-1] for i in range(0, nsp)]
    M_l = [modelB.coef_[i][0:nsp].tolist() for i in range(0, nsp)]

    print("\ninferred params:")
    print("mu_hat/mu/mu_l:")
    print(np.array(mu_h))
    print(np.array(mu))
    print(np.array(mu_l))
    print("\nM_hat/M/M_l:")
    print(np.round(np.array(M_h), decimals=2))
    print("\n", np.array(M))
    print("\n", np.round(np.array(M_l), decimals=2))

    # plot the fit
    yobs_pred = odeint(gLV, y0, times, args=(nsp, mu_h, M_h))
    plt.plot(times, yobs)
    plt.plot(times, yobs_pred, '--')

    # plot the params
    plt.figure()
    plt.stem(np.arange(0, len(mu), dtype="int32"), np.array(mu_h), markerfmt="D")
    plt.stem(np.arange(0, len(mu), dtype="int32"), np.array(mu), markerfmt="X")

    plt.figure()
    plt.stem(np.arange(0, nsp*nsp), np.array(M_h).flatten(), markerfmt="D")
    plt.stem(np.arange(0, nsp*nsp), np.array(M).flatten(), markerfmt="X")

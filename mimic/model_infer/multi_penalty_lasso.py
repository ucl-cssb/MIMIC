import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score


class MultiPenaltyLasso(BaseEstimator, RegressorMixin):
    """Linear regression with non-uniform L1 regularizer.

    Minimizes objective function::

        1 / (2 * n_samples) * ||y - X * w||^2_2 + ||alpha * w||_1

    where::

        alpha = diag(alpha_1, alpha_2, ..., alpha_{num_samples})

    and all alpha are non-zero

    """

    def __init__(self, alpha):
        self.coef_ = None
        self.alpha = alpha

    def fit(self, X, y):
        if sum(self.alpha == 0) == 0:
            self.coef_ = self.non_zero_penalties(X, y)
        else:
            self.coef_ = self.penalised_lasso(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model not fitted")
        return X @ self.coef_.T

    def get_params(self, deep=True) -> dict:
        return {"alpha": self.alpha}

    def non_zero_penalties(self, X, y) -> np.ndarray:
        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        # print(
        #    f'n_samples: {n_samples}, n_features: {n_features}, n_targets: {n_targets}')

        lambda_p = np.diag(self.alpha)

        X_p = X @ np.linalg.inv(lambda_p)

        model = Lasso(fit_intercept=False, max_iter=int(1e6), alpha=1)

        model.fit(X_p, y)

        return model.coef_ @ np.linalg.inv(lambda_p)

    def penalised_lasso(self, X, y) -> np.ndarray:
        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        # print(
        #    f'n_samples: {n_samples}, n_features: {n_features}, n_targets: {n_targets}')

        alpha_zeros_idx = np.nonzero(self.alpha == 0)[0]
        alpha_nonzeros_idx = np.nonzero(self.alpha != 0)[0]

        X_NP = X[:, alpha_zeros_idx]
        print(f'shape of non-penalised: {X_NP.shape}')
        X_P = X[:, alpha_nonzeros_idx]
        print(f'shape of penalised: {X_P.shape}')

        # 1. Project out the non-penalized coefficients: Calculate the residual matrix for the non-penalized
        # coefficients (λ_i=0):
        #   MNP = I − X_NP (X^T_NP X_NP)^−1 X^T_NP
        print((X_NP @ np.linalg.inv(np.transpose(X_NP) @ X_NP)
              @ np.transpose(X_NP)).shape)
        M_NP = np.subtract(np.identity(
            n_samples), X_NP @ np.linalg.inv(np.transpose(X_NP) @ X_NP) @ np.transpose(X_NP))

        # and apply it to the response variable (y → M_NP y)
        y_prime = M_NP @ y

        # 2. Rescale the projected design matrix: Transform the projected design matrix according to the diagonal
        # matrix of the lasso penalties:
        #   M_NP X_P → M_NP X_P Λ^−1_P
        lambda_p = np.diag(self.alpha[alpha_nonzeros_idx])
        projected_X_P = M_NP @ X_P @ np.linalg.inv(lambda_p)

        # 3. Apply a lasso: Lasso regress the projected response variable M_NP y on the projected and scaled design
        # matrix M_NP X_P Λ^−1_P with a regularization penalty λ=1 to obtain estimates for the penalized parameters β̂_P
        if len(alpha_nonzeros_idx) > 0:
            model = Lasso(fit_intercept=False, max_iter=int(1e6), alpha=1)
            model.fit(projected_X_P, y_prime)
            beta_P_hat = model.coef_
        else:
            beta_P_hat = []

        # 4. Apply an ordinary least squares: Finally, ordinary least squares regress the residuals of the response
        # variable y − X_P β̂_P on the non-penalized design matrix X_NP to obtain the non-penalized parameters β̂_NP.
        if len(alpha_zeros_idx) > 0:
            model2 = LinearRegression(fit_intercept=False)
            model2.fit(X_NP, y - X_P @ beta_P_hat)
            beta_NP_hat = model2.coef_
        else:
            beta_NP_hat = []

        coefs = np.empty([n_features, n_targets])
        for i, idx in enumerate(alpha_zeros_idx):
            coefs[idx] = beta_NP_hat[i]
        for i, idx in enumerate(alpha_nonzeros_idx):
            coefs[idx] = beta_P_hat[i]

        return coefs


def fit_alpha_MPLasso(X, y, n_a) -> np.ndarray:

    candidate_alpha_1 = np.logspace(-9, 2, n_a)
    candidate_alpha_2 = np.logspace(-9, 2, n_a)
    candidate_alphas = np.transpose([np.tile(candidate_alpha_1, len(
        candidate_alpha_2)), np.repeat(candidate_alpha_2, len(candidate_alpha_1))])

    candidate_regressors = []
    for a in candidate_alphas:
        # this sets the diag(Lambda) penalties matrix with a[0] penality for M and a[1] penalty for mu
        lambda_p = np.append(np.ones(y.shape[1])*a[0], a[1])
        candidate_regressors.append(MultiPenaltyLasso(alpha=lambda_p))

    # cv = RepeatedKFold(n_splits=5, n_repeats=10)
    cv = KFold(n_splits=5, shuffle=True)

    cv_results = [-cross_val_score(r, X, y, scoring='neg_root_mean_squared_error',
                                   cv=cv, n_jobs=-1) for r in candidate_regressors]

    n_est = np.array([len(x) for x in cv_results])
    cv_means = np.array([np.mean(x) for x in cv_results])
    cv_se = np.array([np.std(x) / np.sqrt(100) for x in cv_results])

    min_i = np.argmin(cv_means)

    print("minimum found: a/error:", candidate_alphas[min_i], cv_means[min_i])

    fig, axs = plt.subplots(1, 2, layout='constrained')
    for i in range(2):
        axs[i].scatter(candidate_alphas[:, i].T,
                       cv_means, marker='o', color='blue')
        axs[i].scatter([candidate_alphas[min_i, i]], [cv_means[min_i]],
                       marker='o', color='red', label='Minimum rule')
        axs[i].legend()
        axs[i].set_xscale('log')
        axs[i].set_xlabel('Log alpha')
        axs[i].set_ylabel('MSE')

    return candidate_alphas[min_i]

import numpy as np
import scipy as sp
import scipy.stats as stats


class BayesianLinearRegression:
    """
    Normal Inverse Gamma Bayesian linear regression class that models the targets y as
    normally distributed around a linear function of the inputs X.
    y | w, σ, X  ~  N(Xw, σ²I)
    The prior and posterior are Normal Inverse Gamma distributions and assume:
    1/σ² ~ Γ(a, b)
    w ~ N(mu, σ²V)
    """

    def __init__(self, mu_0=None, V_0=None, a_0=None, b_0=None):

        self.mu_0 = mu_0
        self.V_0 = V_0
        self.a_0 = a_0
        self.b_0 = b_0

    def fit(self, X, y):
        n_data = X.shape[0]
        n_features = X.shape[1]
        if self.mu_0 is None:
            self.mu_0 = np.zeros((n_features, 1))
        if self.V_0 is None:
            self.V_0 = 1e6 * np.eye(n_features)
        self.V_0_inv = np.linalg.inv(self.V_0)
        if self.a_0 is None:
            self.a_0 = -n_features / 2
        if self.b_0 is None:
            self.b_0 = 0

        V_n_inv = X.T @ X + self.V_0_inv
        V_n = np.linalg.inv(V_n_inv)
        mu_n = V_n @ (self.V_0_inv @ self.mu_0 + X.T @ y)
        a_n = self.a_0 + n_data / 2
        b_n = self.b_0 + 0.5 * (
            y.T @ y + self.mu_0.T @ self.V_0_inv @ self.mu_0 - mu_n.T @ V_n_inv @ mu_n
        )
        self.V_n_inv = V_n_inv
        self.V_n = V_n
        self.mu_n = mu_n
        self.a_n = a_n
        self.b_n = b_n
        return self

    def predict(self, X):
        n_data = X.shape[0]
        df = 2 * self.a_n
        mean = X @ self.mu_n
        dispersion = (self.b_n / self.a_n) * (np.eye(n_data) + X @ self.V_n @ X.T)
        # scale = sp.linalg.sqrtm(dispersion)
        return (
            df,
            mean,
            np.sqrt(np.diag(dispersion)),
        )  # stats.t(df, loc=mean, scale=scale)

    def sample_sigma(self, seed=0):
        np.random.seed(seed)
        ig = stats.invgamma(a=self.a_n, scale=self.b_n)
        # print("mean sigma", np.sqrt(ig.mean()))
        sigma_sq = ig.rvs()
        return np.sqrt(sigma_sq)

    def sample_conditional_w(self, sigma_sq, seed=0):
        np.random.seed(seed)
        multivariate_normal = stats.multivariate_normal(
            mean=self.mu_n.flatten(), cov=sigma_sq * self.V_n
        )
        w = multivariate_normal.rvs().reshape(-1, 1)
        return w

    def sample_functions(self, X, seed=0):
        sigma_sq = self.sample_sigma(seed=seed) ** 2
        w = self.sample_conditional_w(sigma_sq, seed=seed)
        function_values = X @ w
        return function_values, w, sigma_sq


# Taken from https://github.com/tonyduan/conjugate-bayes
# class BayesianLinearRegression(object):
#     """
#     The normal inverse-gamma prior for a linear regression model with unknown
#     variance and unknown relationship. Specifically,
#         1/������������������������������������ ~ ������������������(a, b)
#         ������������������ ~ N(0, ������������������������������������V)
#     Parameters
#     ----------
#     mu: prior for N(mu, v) on the model ������������������
#     v:  prior for N(mu, v) on the model ������������������
#     a:  prior for ������������������(a, b) on the inverse sigma2 of the distribution
#     b:  prior for ������������������(a, b) on the inverse sigma2 of the distribution
#     """
#
#     def __init__(self, mu_0, V_0, a_0, b_0):
#         self.__dict__.update({"mu": mu_0, "v": V_0, "a": a_0, "b": b_0})
#
#     def fit(self, x_tr, y_tr):
#         y_tr = y_tr.flatten()
#         m, _ = x_tr.shape
#         mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ (
#             np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr
#         )
#         v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
#         a_ast = self.a + 0.5 * m
#         b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ np.linalg.inv(
#             np.eye(m) + x_tr @ self.v @ x_tr.T
#         ) @ (y_tr - x_tr @ self.mu.T)
#         self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})
#
#     def predict(self, x_te):
#         scales = np.array([x.T @ self.v @ x for x in x_te]) + 1
#         scales = (self.b / self.a * scales) ** 0.5
#         loc = x_te @ self.mu
#         return 2 * self.a, loc.reshape(-1, 1), scales  # df, loc, scale
#
#     def get_conditional_beta(self, sigma2):
#         return sp.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)
#
#     def get_marginal_sigma2(self):
#         return sp.stats.invgamma(self.a, scale=self.b)
#
#     def get_marginal_beta(self):
#         return (
#             2 * self.a,
#             self.mu,
#             (self.b / self.a * np.diagonal(self.v)) ** 0.5,
#         )  # df, loc, scale

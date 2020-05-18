# Taken from https://github.com/tonyduan/conjugate-bayes
import numpy as np
import scipy as sp


class BayesianLinearRegression(object):
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown relationship. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(0, σ²V)
    Parameters
    ----------
    mu: prior for N(mu, v) on the model β
    v:  prior for N(mu, v) on the model β
    a:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    """

    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ (
            np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr
        )
        v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ np.linalg.inv(
            np.eye(m) + x_tr @ self.v @ x_tr.T
        ) @ (y_tr - x_tr @ self.mu.T)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = np.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return 2 * self.a, x_te @ self.mu, scales  # df, loc, scale

    def get_conditional_beta(self, sigma2):
        return sp.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return sp.stats.invgamma(self.a, scale=self.b)

    def get_marginal_beta(self):
        return (
            2 * self.a,
            self.mu,
            (self.b / self.a * np.diagonal(self.v)) ** 0.5,
        )  # df, loc, scale

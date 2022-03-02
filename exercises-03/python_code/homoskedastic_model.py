import numpy as np
from scipy.stats import multivariate_t
from scipy import linalg


class HomoskedasticModel:
    def __init__(self, X, y, lam=None, K=None, m=None, d=None, eta=None, fit_intercept=True):
        """

        Parameters
        ----------
        y ~ N(X@beta, (omega lambda)^(-1))
        beta ~ N(m, (omega K)^(-1))
        omega ~ gamma(d/2, eta/2)

        X: (nxp) data
        y: (n) response
        lam: (nxn) precision
        K: (pxp) precision
        m: (p) mean
        d: scalar
        eta: scalar
        fit_intercept: bool
        """
        self.fit_intercept = fit_intercept
        self.X = self._initialize_X(X)
        self.y = y
        self.lam = self._initialize_lam(lam)
        self.K = self._initialize_K(K)
        self.m = self._initialize_m(m)
        self.d = self._initialize_d(d)
        self.eta = self._initialize_eta(eta)
        self.n = self.X.shape[0]
        self.coefs = None

    def _initialize_X(self, X):
        if self.fit_intercept:
            return self._make_intercept(X)
        else:
            return X

    @staticmethod
    def _make_intercept(X):
        """add intercept"""
        column_ones = np.ones(X.shape[0])[:, None]
        bigger_X = np.hstack((column_ones, X))
        return bigger_X

    def _initialize_lam(self, lam):
        if lam is not None:
            return lam
        return np.identity(self.X.shape[0])

    def _initialize_K(self, K):
        if K is not None:
            return K
        return np.identity(self.X.shape[1]) * .001

    def _initialize_m(self, m):
        if m is not None:
            return m
        return np.zeros(self.X.shape[1])

    @staticmethod
    def _initialize_d(d):
        if d is not None:
            return d
        return 1

    @staticmethod
    def _initialize_eta(eta):
        if eta is not None:
            return eta
        return 1

    def _calculate_K_star(self):
        K_star = self.X.T @ self.lam @ self.X + self.K
        return K_star

    def _calculate_m_star(self):
        K_star = self._calculate_K_star()
        m_star = np.linalg.inv(K_star) @ (self.X.T @ self.lam @ self.y + self.K @ self.m)
        return m_star

    def _calculate_d_star(self):
        return self.d + self.n

    def _calculate_eta_star(self):
        m_star = self._calculate_m_star()
        K_star = self._calculate_K_star()
        eta_star = self.eta + self.y.T @ self.lam @ self.y + self.m.T @ self.K @ self.m - m_star.T @ K_star @ m_star
        return eta_star

    def calculate_posterior_distribution(self):
        """ returns P(beta|y) which is student t"""
        mean = self._calculate_m_star()
        d_star = self._calculate_d_star()
        eta_star = self._calculate_eta_star()
        K_star = self._calculate_K_star()
        shape = np.linalg.inv(K_star) * eta_star / d_star
        t_dist = multivariate_t(loc=mean, shape=shape, df=d_star)
        return t_dist

    def fit(self):
        """calculate coefs, best estimate is mean of t dist"""
        self.coefs = self._calculate_m_star()

    def predict(self, X):
        X_pred = self._initialize_X(X)
        y = X_pred@self.coefs
        return y

    def calculate_confidence_interval(self, alpha=0.05, num_samples=10000):
        """sample from t distribution, order samples and take alpha bounds"""
        dist = self.calculate_posterior_distribution()
        sorted_matrix = np.sort(dist.rvs(num_samples), axis=0)
        lower_index = int(num_samples * alpha / 2) - 1
        higher_index = int(num_samples * (1-alpha / 2)) - 1
        conf_interval = sorted_matrix[lower_index, :], sorted_matrix[higher_index, :]
        return conf_interval


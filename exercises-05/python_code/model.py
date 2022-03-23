import numpy as np
from scipy.stats import multivariate_normal, gamma
from tqdm import tqdm


class Model:
    def __init__(self, X_groups, y_groups, total_observations, n_iter=1000):
        self.X_groups = X_groups
        self.y_groups = y_groups
        self.n_stores = len(self.X_groups)
        self.n_par = 4
        self.N = total_observations
        self.tau_squared = 100
        self.sigma_squared = 100
        self.gamma = np.ones(self.n_par)
        self.betas = np.ones((self.n_par, len(X_groups)))
        self.n_iter = n_iter
        self.traces = {'betas': np.zeros([self.n_iter, self.n_par, len(X_groups)]), 'gamma': np.zeros((self.n_iter, self.n_par)), 'sigma_squared': np.zeros(self.n_iter), 'tau_squared': np.zeros(self.n_iter)}

    def _update_betas(self):
        for store_number in range(self.n_stores):
            K_inv = np.linalg.inv(1 / self.sigma_squared * (self.X_groups[store_number].T @ self.X_groups[
                store_number] + 1 / self.tau_squared * np.identity(self.n_par)))
            m = 1 / self.sigma_squared * K_inv @ (
                    self.X_groups[store_number].T @ self.y_groups[store_number] + 1 / self.tau_squared * self.gamma)
            self.betas[:, store_number] = multivariate_normal(m, K_inv).rvs()

    def _update_sigma_squared(self):
        alpha = (self.N + self.n_stores) / 2
        beta = sum([1 / 2 * (self.y_groups[store_number] - self.X_groups[store_number]@self.betas[:,store_number]).T @ (
                self.y_groups[store_number] - self.X_groups[store_number]@self.betas[:,store_number]) + 1 / self.tau_squared * (
                            self.betas[:, store_number] - self.gamma).T @ (self.betas[:, store_number] - self.gamma)
                    for store_number in range(self.n_stores)])
        self.sigma_squared = 1 / gamma(alpha, 1 / beta).rvs()

    def _update_gamma(self):
        K_inv = np.linalg.inv(self.n_stores * np.identity(self.n_par) / (self.sigma_squared * self.tau_squared))
        m = K_inv @ np.sum(self.betas, axis=1) / (self.sigma_squared * self.tau_squared)
        self.gamma = multivariate_normal(m, K_inv).rvs()

    def _update_tau_squared(self):
        alpha = self.n_stores + 1 / 2
        beta = sum([0.5 * (
                (1 / self.sigma_squared * (self.betas[:, store_number] - self.gamma).T @ (self.betas[:, store_number] - self.gamma)) + 1) for
                    store_number in range(self.n_stores)])
        self.tau_squared = 1 / gamma(alpha, 1 / beta).rvs()

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas
        self.traces['gamma'][it, :] = self.gamma
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared

    def _discard_burn(self):
        for key in self.traces.keys():
            self.traces[key] = self.traces[key][100:]

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_gamma()
            self._update_sigma_squared()
            self._update_betas()
            self._update_tau_squared()
            self._update_traces(it)
        self._discard_burn()

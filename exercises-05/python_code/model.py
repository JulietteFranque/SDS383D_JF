import numpy as np
from scipy.stats import multivariate_normal, gamma
from tqdm import tqdm
import matplotlib.pyplot as plt


class Model:
    def __init__(self, X_groups, y_groups, total_observations, n_iter=1000, burn=100):
        self.X_groups = X_groups
        self.y_groups = y_groups
        self.n_stores = len(self.X_groups)
        self.n_par = 4
        self.N = total_observations
        self.n_iter = n_iter
        self.traces = {'betas': np.zeros([self.n_iter, self.n_par, len(X_groups)]),
                       'gamma': np.zeros((self.n_iter, self.n_par)), 'sigma_squared': np.zeros(self.n_iter),
                       'Lambda': np.zeros((self.n_iter, self.n_par))}
        self.burn = burn
        self.Lambda_matrix = np.diag([1] * 4)
        self.sigma_squared = 1
        self.gamma = np.ones(self.n_par)
        self.betas = np.ones((self.n_par, len(X_groups)))

    def _update_betas(self):
        for store in range(self.n_stores):
            K_inv = np.linalg.inv(self.X_groups[store].T @ self.X_groups[store] / self.sigma_squared +
                                  self.inverse_Lambda / self.sigma_squared)
            m = K_inv @ (self.X_groups[store].T @ self.y_groups[store] / self.sigma_squared +
                         self.inverse_Lambda @ self.gamma / self.sigma_squared)
            self.betas[:, store] = multivariate_normal.rvs(m, K_inv)

    def _update_sigma_squared(self):
        alpha = (self.N + self.n_stores) / 2
        beta = 1 / 2 * sum(
            [(self.y_groups[store_number] - self.X_groups[store_number] @ self.betas[:, store_number]).T @ (
                    self.y_groups[store_number] - self.X_groups[store_number] @ self.betas[:, store_number]) + (
                     self.betas[:, store_number] - self.gamma).T @ self.inverse_Lambda @ (
                     self.betas[:, store_number] - self.gamma) for store_number in range(self.n_stores)])
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta)

    def _update_gamma(self):
        K_inv = np.linalg.inv(self.n_stores * self.inverse_Lambda / self.sigma_squared)
        m = K_inv @ self.inverse_Lambda @ np.sum(self.betas, axis=1) / self.sigma_squared
        self.gamma = multivariate_normal(m, K_inv).rvs()

    def _update_Lambda_matrix(self):
        alpha = (self.n_stores + 1) / 2
        beta = 0.5 * (((self.betas - self.gamma[:, None]) ** 2).sum(axis=1) / self.sigma_squared + 1)
        Lambda_ii = 1 / gamma.rvs(a=alpha, scale=1 / beta)
        self.Lambda_matrix = np.diag(Lambda_ii)
        self.inverse_Lambda = np.linalg.inv(self.Lambda_matrix)

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas
        self.traces['gamma'][it, :] = self.gamma
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['Lambda'][it, :] = np.diag(self.Lambda_matrix)

    def _discard_burn(self):
        keys = list(self.traces.keys())
        for key in keys:
            self.traces[key] = self.traces[key][self.burn:]

    def plot_all_traces(self):
        keys = list(self.traces.keys())
        plt.figure(figsize=(8, 9))
        for i, key in enumerate(keys):
            plt.subplot(len(keys), 1, i + 1)
            plt.title(key, fontsize=16)
            plt.xticks([])
            plt.yticks([])
            if key == 'betas':
                betas_traces = self.traces['betas']
                betas_traces = betas_traces.reshape(betas_traces.shape[0],
                                                    betas_traces.shape[1] * betas_traces.shape[2])
                plt.plot(betas_traces, color='dodgerblue', alpha=.5)
            else:
                plt.plot(self.traces[key], color='dodgerblue', alpha=.7)

    def plot_all_histograms(self):
        self._plot_beta_histograms()
        self._plot_sigma_histograms()
        self._plot_gamma_histogram()
        self._plot_tau_squared_histograms()

    def _plot_beta_histograms(self):
        for j in range(self.n_par):
            for i in range(self.n_stores):
                plt.hist(self.traces['betas'][:, j, i], density=True, alpha=.5, bins=100)
            plt.title(f'beta {j+1}', fontsize=16)
            plt.figure()

    def _plot_sigma_histograms(self):
        plt.hist(self.traces['sigma_squared'], density=True, alpha=.5, bins=100)
        plt.title('sigma_squared', fontsize=16)
        plt.figure()

    def _plot_gamma_histogram(self):
        for i in range(self.n_par):
            plt.hist(self.traces['gamma'][:, i], density=True, alpha=.5, bins=100)
        plt.title('gammas', fontsize=16)
        plt.figure()

    def _plot_tau_squared_histograms(self):
        for i in range(self.n_par):
            plt.hist(self.traces['Lambda'][:, i], density=True, alpha=.5, bins=100)
        plt.title('Lambda', fontsize=16)
        plt.figure()

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_Lambda_matrix()
            self._update_gamma()
            self._update_sigma_squared()
            self._update_betas()
            self._update_traces(it)
        self._discard_burn()

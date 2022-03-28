import numpy as np
from scipy.stats import multivariate_normal, gamma, truncnorm
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})


class Model:
    def __init__(self, X, y, n_iter=1000, burn=100):
        self.X = X
        self.y = y
        self._burn = burn
        self._n_betas = self.X[0].shape[1]
        self._n_groups = len(self.X)
        self.n_iter = n_iter
        self._lower_bounds, self._upper_bounds = self._get_trunc_normal_params()
        self._tau_squared = self._initialize_tau_squared()
        self._betas = self._initialize_betas()
        self._m = self._initialize_m()
        self._z = self._initialize_z()
        self.traces = {'betas': np.zeros((self.n_iter, self._n_betas, self._n_groups)),
                       'tau_squared': np.zeros(self.n_iter),
                       'm': np.zeros((self.n_iter, self._n_betas))}

    @staticmethod
    def _initialize_tau_squared():
        return 1

    def _initialize_betas(self):
        return np.ones((self._n_betas, self._n_groups))

    def _initialize_m(self):
        return np.ones(self._n_betas)

    def _initialize_z(self):
        return [np.random.choice([0, 1], size=len(x)) for x in self.y]

    def _get_trunc_normal_params(self):
        lower_bounds = []
        upper_bounds = []
        for group in range(self._n_groups):
            lower_bound = [-np.inf if x == 0 else 0 for x in self.y[group]]
            upper_bound = [np.inf if x == 1 else 0 for x in self.y[group]]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        return lower_bounds, upper_bounds

    def _discard_burn(self):
        keys = list(self.traces.keys())
        for key in keys:
            self.traces[key] = self.traces[key][self._burn:]

    def _update_betas(self):
        for group in range(self._n_groups):
            cov = np.linalg.inv(self._tau_squared * np.eye(self._n_betas) + self.X[group].T @ self.X[group])
            mean = cov @ (1 / self._tau_squared * self._m + self.X[group].T @ self._z[group])
            self._betas[:, group] = multivariate_normal.rvs(mean=mean, cov=cov)

    def _update_tau_squared(self):
        alpha = (self._n_groups + 1) / 2
        beta = 0.5 * (((self._betas - self._m[:, None]) ** 2).sum() + 1)
        self._tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta)

    def _update_m(self):
        cov = self._tau_squared / self._n_groups * np.eye(self._n_betas)
        mean = cov / self._tau_squared @ self._betas.sum(axis=1)
        self._m = multivariate_normal.rvs(mean=mean, cov=cov)

    def _update_z(self):
        self._z = []
        for group in range(self._n_groups):
            self._z.append(truncnorm.rvs(self._lower_bounds[group], self._upper_bounds[group], loc=0, scale=1))

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_tau_squared()
            self._update_betas()
            self._update_m()
            self._update_z()
            self._update_traces(it)
        self._discard_burn()

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self._betas
        self.traces['m'][it, :] = self._m
        self.traces['tau_squared'][it] = self._tau_squared

    def _plot_beta_histograms(self):
        for j in range(self._n_betas):
            for i in range(self._n_groups):
                plt.hist(self.traces['betas'][:, j, i], density=True, alpha=.5, bins=50)
            plt.title(f'beta {j + 1}', fontsize=16)
            plt.savefig(f'beta{j + 1}.png', dpi=600)
            plt.figure()

    def _plot_tau_histograms(self):
        plt.hist(self.traces['tau_squared'], density=True, alpha=.5, bins=50)
        plt.title('tau_squared', fontsize=16)
        plt.savefig('tau_sq.png', dpi=600)
        plt.figure()

    def _plot_m_histogram(self):
        for i in range(self._n_betas):
            plt.hist(self.traces['m'][:, i], density=True, alpha=.5, bins=50)
        plt.title('m', fontsize=16)
        plt.savefig('m.png', dpi=600)
        plt.figure()

    def plot_all_histograms(self):
        self._plot_m_histogram()
        self._plot_tau_histograms()
        self._plot_beta_histograms()

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

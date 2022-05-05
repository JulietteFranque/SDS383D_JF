import numpy as np
from scipy.stats import multivariate_normal, gamma, uniform, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


class GibbsSampler:
    def __init__(self, X, y, n_iter=1000, burn=500, betas_start=1, sigmas_start=1, mu_start=1,
                 sigmas_diag_start=1):
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.burn = burn
        self.n_dept = len(self.X)
        self.n_par = self.X[0].shape[1]
        self.number_obs_in_each_dept = np.array([X_dept.shape[0] for X_dept in self.X])
        self.betas, self.sigmas_diag, self.mu, self.sigmas_squared = self._initialize_parameters(betas_start, sigmas_start, mu_start, sigmas_diag_start)
        self.traces = self._initialize_traces()

    def _initialize_parameters(self, betas_start, sigmas_start, mu_start, sigmas_diag_start):
        betas = np.ones([self.n_par, len(self.X)]) * betas_start
        sigmas_diag = np.ones(self.n_par) * sigmas_diag_start
        mu = np.ones(self.n_par) * mu_start
        sigmas_squared = np.ones(self.n_dept) * sigmas_start
        return betas, sigmas_diag, mu, sigmas_squared

    def _initialize_traces(self):
        traces = {'betas': np.ones([self.n_iter, self.n_par, self.n_dept]) * np.nan,
                  'mu': np.ones((self.n_iter, self.n_par)) * np.nan,
                  'sigmas_diag': np.ones((self.n_iter, self.n_par)) * np.nan,
                  'bandwidths': np.ones((self.n_iter, self.n_dept)) * np.nan,
                  'sigmas_squared': np.ones((self.n_iter, self.n_dept)) * np.nan
                  }
        return traces

    def _update_sigmas_squared(self):
        alpha = np.ones(self.n_dept) * self.number_obs_in_each_dept / 2
        beta = 0.5 * np.array([(self.y[dept] - self.X[dept]@self.betas[:, dept]).T @(self.y[dept] - self.X[dept]@self.betas[:, dept]) for dept in
                         range(self.n_dept)])
        sigmas = 1 / gamma(a=alpha, scale=1 / beta).rvs()
        self.sigmas_squared = sigmas

    def _update_mu(self):
        mean = np.mean(self.betas, axis=1)
        cov = 1 / self.n_dept * np.diag(self.sigmas_diag)
        self.mu = multivariate_normal(mean, cov).rvs()

    def _update_sigmas_diag(self):
        alpha = np.ones(self.n_par) * (self.n_dept + 1) / 2
        beta = 0.5 * (((self.betas.T - self.mu)**2).sum(axis=0) + 1)
        self.sigmas_diag = 1 / gamma(a=alpha, scale=1/beta).rvs()

    def _update_betas(self):
        for dept_idx in range(self.n_dept):
            sigma_mat_inv = np.linalg.inv(np.diag(self.sigmas_diag))
            cov_mat_inv = 1/self.sigmas_squared[dept_idx] * np.eye(self.number_obs_in_each_dept[dept_idx])
            cov = np.linalg.inv(self.X[dept_idx].T @ cov_mat_inv @ self.X[dept_idx] + sigma_mat_inv)
            mean = cov @ (self.X[dept_idx].T @ cov_mat_inv @ self.y[dept_idx] + sigma_mat_inv @ self.mu)
            self.betas[:, dept_idx] = multivariate_normal(mean, cov).rvs()

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_sigmas_diag()
            self._update_sigmas_squared()
            self._update_mu()
            self._update_betas()
            self._update_traces(it)
        self._discard_burn()

    def _discard_burn(self):
        keys = list(self.traces.keys())
        for key in keys:
            self.traces[key] = self.traces[key][self.burn:]

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas
        self.traces['mu'][it, :] = self.mu
        self.traces['sigmas_squared'][it, :] = self.sigmas_squared
        self.traces['sigmas_diag'][it, :] = self.sigmas_diag

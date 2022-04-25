import numpy as np
from scipy.stats import multivariate_normal, gamma, uniform, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm


class GibbsSampler:
    def __init__(self, X, y, time_vecs, n_iter=1000, burn=500):
        self.X = X
        self.y = y
        self.time_vecs = time_vecs
        self.n_dept = len(self.X)
        self.n_par = self.X[0].shape[1]
        self.number_obs_in_each_dept = np.array([X_dept.shape[0] for X_dept in self.X])
        self.betas = np.ones([self.n_par, len(self.X)])
        self.taus = np.ones(self.n_par)
        self.mu = np.ones(self.n_par)
        self.sigmas = np.ones(self.n_dept)
        self.n_iter = n_iter
        self.bandwidths = np.ones(self.n_dept)
        self.tau_sq_1s = np.ones(self.n_dept) * 10
        self.bandwidths_posteriors = np.zeros(self.n_dept)
        self.tau_sq_posteriors = np.zeros(self.n_dept)
        self.corr_mats = self._initialize_cov_mats()
        self.sigmas_squared = np.ones(self.n_dept)
        self.sigmas_squared_posteriors = np.zeros(self.n_dept)
        self.uncorr_mats = [self.sigmas_squared[n] * np.eye(n_obs) for n, n_obs in
                            enumerate(self.number_obs_in_each_dept)]

        self.traces = {'betas': np.zeros([self.n_iter, self.n_par, self.n_dept]),
                       'mu': np.zeros((self.n_iter, self.n_par)),
                       'taus': np.zeros((self.n_iter, self.n_par)),
                       'bandwidths': np.zeros((self.n_iter, self.n_dept)),
                       'sigmas_squared': np.zeros((self.n_iter, self.n_dept)),
                       'tau_sq_1s': np.zeros((self.n_iter, self.n_dept)),
                       'corr_mats': [
                           np.ones([self.number_obs_in_each_dept[n], self.number_obs_in_each_dept[n], self.n_iter]) for
                           n in range(self.n_dept)]}
        self.burn = burn

    def _initialize_cov_mats(self):
        cov_mats = [self._calculate_exp_covariance_function(self.time_vecs[n], self.time_vecs[n], self.bandwidths[n],
                                                            self.tau_sq_1s[n]) for n in range(self.n_dept)]
        return cov_mats

    def _update_mu(self):
        mean = 1 / self.n_dept * np.mean(self.betas, axis=1)
        cov = 1 / self.n_dept * np.diag(self.taus)
        self.mu = multivariate_normal(mean, cov).rvs()

    def _update_cov_mats(self):
        self.cov_mats = [self.uncorr_mats[n] + self.corr_mats[n] for n in range(self.n_dept)]

    def _update_uncorr_mats(self):
        for dept in range(self.n_dept):
            new_sig_squared = norm(self.sigmas_squared[dept], 1).rvs()
            new_cov = self.corr_mats[dept] + new_sig_squared * np.eye(self.number_obs_in_each_dept[dept])
            new_posterior = self._calculate_sigmas_posterior(dept, new_sig_squared, new_cov)
            log_ratio = self._calculate_log_metropolis_hastings_ratio(new_posterior,
                                                                      self.sigmas_squared_posteriors[dept])
            accept = self._accept_or_reject(log_ratio)
            if accept:
                self.sigmas_squared[dept] = new_sig_squared
                self.sigmas_squared_posteriors[dept] = new_posterior
                self.uncorr_mats[dept] = new_sig_squared * np.eye(self.number_obs_in_each_dept[dept])

    def _update_taus(self):
        alpha = (self.n_dept + 1) / 2
        beta = 0.5 * (np.sum((self.betas - self.mu[:, None]) ** 2, axis=1) + 1)
        self.taus = 1 / gamma(a=alpha, scale=1 / beta).rvs()

    def _update_tau_sq_1s_and_corr_mat(self):
        for dept in range(self.n_dept):
            new_tau_sq = norm(self.tau_sq_1s[dept], 1).rvs()
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=new_tau_sq) + self.uncorr_mats[dept]
            new_posterior = self._calculate_tau_sq_1_posterior(dept, new_tau_sq, new_cov)
            log_ratio = self._calculate_log_metropolis_hastings_ratio(new_posterior, self.tau_sq_posteriors[dept])
            accept = self._accept_or_reject(log_ratio)
            if accept:
                self.tau_sq_1s[dept] = new_tau_sq
                self.tau_sq_posteriors[dept] = new_posterior
                self.corr_mats[dept] = new_cov

    def _update_bandwidths(self):
        for dept in range(self.n_dept):
            new_bandwidth = norm(self.bandwidths[dept], 1).rvs()
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=new_bandwidth,
                                                              tau_sq_1=self.tau_sq_1s[dept]) + self.uncorr_mats[dept]
            new_posterior = self._calculate_bandwidth_posterior(dept, new_bandwidth, new_cov)
            log_ratio = self._calculate_log_metropolis_hastings_ratio(new_posterior, self.bandwidths_posteriors[dept])
            accept = self._accept_or_reject(log_ratio)
            if accept:
                self.bandwidths[dept] = new_bandwidth
                self.bandwidths_posteriors[dept] = new_posterior

    def _calculate_bandwidth_posterior(self, dept, sigma_sq, cov):
        likelihood = self._draw_likelihood(self.X[dept], self.betas[:, dept], self.y[dept], cov)
        prior = uniform.pdf(sigma_sq, loc=1, scale=100)
        return likelihood * prior

    def _calculate_sigmas_posterior(self, dept, bandwidth, cov):
        likelihood = self._draw_likelihood(self.X[dept], self.betas[:, dept], self.y[dept], cov)
        prior = uniform.pdf(bandwidth, loc=1, scale=100)
        return likelihood * prior

    def _calculate_tau_sq_1_posterior(self, dept, tau_sq_1, cov):
        likelihood = self._draw_likelihood(self.X[dept], self.betas[:, dept], self.y[dept], cov)
        prior = uniform.pdf(tau_sq_1, loc=1, scale=100)
        return likelihood * prior

    @staticmethod
    def _calculate_log_metropolis_hastings_ratio(new_posterior, old_posterior):
        ratio = np.log(new_posterior / old_posterior)
        return ratio

    @staticmethod
    def _accept_or_reject(log_ratio):
        if log_ratio > 0:
            return True
        else:
            r = uniform.rvs()
            if np.log(r) < log_ratio:
                return True
            else:
                return False

    @staticmethod
    def _draw_likelihood(X_dept, betas_dept, y_dept, cov):
        likelihood = multivariate_normal.pdf(y_dept, mean=X_dept @ betas_dept, cov=cov)
        return likelihood

    @staticmethod
    def _calculate_exp_covariance_function(x_1, x_2, bandwidth, tau_sq_1, tau_sq_2=1e-6):
        distance = distance_matrix(x_1.reshape(-1, 1), x_2.reshape(-1, 1))
        cov = tau_sq_1 * np.exp(-1 / 2 * (distance / bandwidth) ** 2) + tau_sq_2 * np.eye(x_1.shape[0], x_2.shape[0])
        return cov

    def _update_betas(self):
        for dept_idx in range(self.n_dept):
            tau_mat_inv = np.linalg.inv(np.diag(self.taus))
            cov_mat_inv = np.linalg.inv(self.cov_mats[dept_idx])
            cov = np.linalg.inv(self.X[dept_idx].T @ cov_mat_inv @ self.X[dept_idx] + tau_mat_inv)
            mean = cov @ (self.X[dept_idx].T @ cov_mat_inv @ self.y[dept_idx] + tau_mat_inv @ self.mu)
            self.betas[:, dept_idx] = multivariate_normal(mean, cov).rvs()

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            try:
                self._update_bandwidths()
                self._update_tau_sq_1s_and_corr_mat()
                self._update_uncorr_mats()
                self._update_cov_mats()
                self._update_taus()
                self._update_mu()
                self._update_betas()
                self._update_traces(it)
            except ValueError:
                pass
        self._discard_burn()

    def _discard_burn(self):
        keys = list(self.traces.keys())
        keys.remove('corr_mats')
        for key in keys:
            self.traces[key] = self.traces[key][self.burn:]

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas
        self.traces['mu'][it, :] = self.mu
        self.traces['sigmas_squared'][it, :] = self.sigmas_squared
        self.traces['taus'][it, :] = self.taus
        self.traces['bandwidths'][it, :] = self.bandwidths
        self.traces['tau_sq_1s'][it, :] = self.tau_sq_1s
        for n in range(self.n_dept):
            self.traces['corr_mats'][n][:, :, it] = self.corr_mats[n]

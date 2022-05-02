import numpy as np
from scipy.stats import multivariate_normal, gamma, uniform, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


class GibbsSampler:
    def __init__(self, X, y, time_vecs, n_iter=1000, burn=500, betas_start=1, sigmas_start=1, mu_start=1,
                 sigmas_diag_start=1, bandwidth_start=10, tau_sq_1_start=20, f_start=10):
        self.X = X
        self.y = y
        self.time_vecs = time_vecs
        self.n_iter = n_iter
        self.burn = burn
        self.n_dept = len(self.X)
        self.n_par = self.X[0].shape[1]
        self.number_obs_in_each_dept = np.array([X_dept.shape[0] for X_dept in self.X])
        self.betas, self.sigmas_diag, self.mu, self.bandwidths, self.sigmas_squared, self.bandwidths, self.tau_sq_1s, self.f, self.gp_mats = self._initialize_parameters(
            betas_start, sigmas_start, mu_start, sigmas_diag_start, bandwidth_start, tau_sq_1_start, f_start)
        self.accept_tau = np.zeros(self.n_dept, dtype=bool)
        self.accept_bandwidth = np.zeros(self.n_dept, dtype=bool)
        self.traces = self._initialize_traces()

    def _initialize_parameters(self, betas_start, sigmas_start, mu_start, sigmas_diag_start, bandwidth_start,
                               tau_sq_1_start, f_start):
        betas = np.ones([self.n_par, len(self.X)]) * betas_start
        sigmas_diag = np.ones(self.n_par) * sigmas_diag_start
        mu = np.ones(self.n_par) * mu_start
        bandwidths = np.ones(self.n_dept) * bandwidth_start
        sigmas_squared = np.ones(self.n_dept) * sigmas_start
        tau_sq_1s = np.ones(self.n_dept) * tau_sq_1_start
        f = [np.ones(self.number_obs_in_each_dept[n]) * f_start for n in range(self.n_dept)]
        gp_mats = [np.eye(self.number_obs_in_each_dept[dept]) for dept in
                   range(self.n_dept)]
        return betas, sigmas_diag, mu, bandwidths, sigmas_squared, bandwidths, tau_sq_1s, f, gp_mats

    def _initialize_traces(self):
        traces = {'betas': np.ones([self.n_iter, self.n_par, self.n_dept]) * np.nan,
                  'mu': np.ones((self.n_iter, self.n_par)) * np.nan,
                  'taus': np.ones((self.n_iter, self.n_par)) * np.nan,
                  'bandwidths': np.ones((self.n_iter, self.n_dept)) * np.nan,
                  'sigmas_squared': np.ones((self.n_iter, self.n_dept)) * np.nan,
                  'tau_sq_1s': np.ones((self.n_iter, self.n_dept)) * np.nan,
                  'f': [
                      np.ones([self.number_obs_in_each_dept[n], self.n_iter]) * np.nan for
                      n in range(self.n_dept)],
                  'accept_bandwidth': np.zeros((self.n_iter, self.n_dept), dtype=bool),
                  'accept_tau_sq': np.zeros((self.n_iter, self.n_dept), dtype=bool)}
        return traces

    def _update_cov_mats(self):
        alpha = self.number_obs_in_each_dept / 2
        beta = np.array([0.5 * (self.y[i] - self.f[i]).T @ (
                self.y[i] - self.f[i]) for i in
                         range(self.n_dept)])
        sigmas = 1 / gamma(a=alpha, scale=1 / beta).rvs()
        self.sigmas_squared = sigmas
        self.cov_mats = [sigmas[n] * np.eye(n_obs) for n, n_obs in enumerate(self.number_obs_in_each_dept)]

    def _update_mu(self):
        mean = 1 / self.n_dept * np.mean(self.betas, axis=1)
        cov = 1 / self.n_dept * np.diag(self.sigmas_diag)
        self.mu = multivariate_normal(mean, cov, allow_singular=True).rvs()

    def _update_taus(self):
        alpha = (self.n_dept + 1) / 2
        beta = 0.5 * (np.sum((self.betas - self.mu[:, None]) ** 2, axis=1) + 1)
        self.sigmas_diag = 1 / gamma(a=alpha, scale=1 / beta).rvs()

    def _update_f(self):
        for dept in range(self.n_dept):
            cov = np.linalg.inv(
                1 / self.sigmas_squared[dept] * np.eye(self.number_obs_in_each_dept[dept]) + np.linalg.inv(
                    self.gp_mats[dept]))
            m = cov @ (np.linalg.inv(self.gp_mats[dept]) @ self.X[dept] @ self.betas[:, dept] + 1 / self.sigmas_squared[
                dept] * self.y[dept])
            self.f[dept] = multivariate_normal(mean=m, cov=cov, allow_singular=True).rvs()

    def _update_tau_sq_1s_and_corr_mat(self):
        for dept in range(self.n_dept):
            new_log_tau_sq1 = norm(loc=np.log(self.tau_sq_1s[dept]), scale=.1).rvs()
            new_tau_sq_1 = np.exp(new_log_tau_sq1)
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=new_tau_sq_1)
            new_prior = uniform.pdf(new_tau_sq_1, loc=1, scale=9999)
            new_likelihood = self._draw_likelihood(dept, new_cov)
            old_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=self.tau_sq_1s[dept])
            old_prior = uniform.pdf(self.tau_sq_1s[dept], loc=1, scale=9999)
            old_likelihood = self._draw_likelihood(dept, old_cov)
            accept = self._accept_or_reject(new_prior, new_likelihood, old_prior, old_likelihood)
            if accept:
                self.tau_sq_1s[dept] = new_tau_sq_1
                self.gp_mats[dept] = new_cov
                self.accept_tau[dept] = True

    def _update_bandwidths(self):
        for dept in range(self.n_dept):
            new_log_bandwidth = norm(loc=np.log(self.bandwidths[dept]), scale=.1).rvs()
            new_bandwidth = np.exp(new_log_bandwidth)
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=new_bandwidth,
                                                              tau_sq_1=self.tau_sq_1s[dept])
            new_prior = uniform.pdf(new_bandwidth, loc=1, scale=99)
            new_likelihood = self._draw_likelihood(dept, new_cov)
            old_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=self.tau_sq_1s[dept])

            old_prior = uniform.pdf(self.bandwidths[dept], loc=1, scale=99)
            old_likelihood = self._draw_likelihood(dept, old_cov)
            accept = self._accept_or_reject(new_prior, new_likelihood, old_prior, old_likelihood)
            if accept:
                self.bandwidths[dept] = new_bandwidth
                self.accept_bandwidth[dept] = True

    @staticmethod
    def _accept_or_reject(new_prior, new_likelihood, old_prior, old_likelihood):
        r = uniform.rvs()
        mcmc_ratio = np.log(new_likelihood) + np.log(new_prior) - np.log(old_likelihood) - np.log(old_prior)
        if mcmc_ratio is None:
            return False
        if mcmc_ratio > r:
            return True
        else:
            return False

    def _draw_likelihood(self, dept, cov):
        likelihood = multivariate_normal(mean=self.X[dept] @ self.betas[:, dept], cov=cov, allow_singular=True).pdf(
            self.f[dept])
        return likelihood

    @staticmethod
    def _calculate_exp_covariance_function(x_1, x_2, bandwidth, tau_sq_1, tau_sq_2=1e-6):
        distance = distance_matrix(x_1.reshape(-1, 1), x_2.reshape(-1, 1))
        cov = tau_sq_1 * np.exp(-1 / 2 * (distance / bandwidth) ** 2) + tau_sq_2 * np.eye(x_1.shape[0], x_2.shape[0])
        return cov

    def _update_betas(self):
        for dept_idx in range(self.n_dept):
            sigma_mat_inv = np.linalg.inv(np.diag(self.sigmas_diag))
            cov_mat_inv = np.linalg.inv(self.gp_mats[dept_idx])
            cov = np.linalg.inv(self.X[dept_idx].T @ cov_mat_inv @ self.X[dept_idx] + sigma_mat_inv)
            mean = cov @ (self.X[dept_idx].T @ cov_mat_inv @ self.f[dept_idx] + sigma_mat_inv @ self.mu)
            self.betas[:, dept_idx] = multivariate_normal(mean, cov, allow_singular=True).rvs()

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            try:
                self._update_bandwidths()
                self._update_tau_sq_1s_and_corr_mat()
                self._update_f()
                self._update_cov_mats()
                self._update_taus()
                self._update_mu()
                self._update_betas()
                self._update_traces(it)
            except:
                print('PROBLEM')

        # except:
        #  print('PROBLEM')

        self._discard_burn()

    def _discard_burn(self):
        keys = list(self.traces.keys())
        keys.remove('f')
        for key in keys:
            self.traces[key] = self.traces[key][self.burn:]

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.betas
        self.traces['mu'][it, :] = self.mu
        self.traces['sigmas_squared'][it, :] = self.sigmas_squared
        self.traces['taus'][it, :] = self.sigmas_diag
        self.traces['bandwidths'][it, :] = self.bandwidths
        self.traces['tau_sq_1s'][it, :] = self.tau_sq_1s
        self.traces['accept_bandwidth'][it, :] = self.accept_bandwidth
        self.traces['accept_tau_sq'][it, :] = self.accept_tau

        for n in range(self.n_dept):
            self.traces['f'][n][:, it] = self.f[n]

import numpy as np
from scipy.stats import multivariate_normal, gamma, uniform, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


class GibbsSampler:
    def __init__(self, X, y, time_vecs, n_iter=150000, burn=0, betas_start=1, sigmas_start=1, mu_start=1,
                 sigmas_diag_start=1, bandwidth_start=10, tau_sq_1_start=20, f_start=10, sigmas_proposal=[1,2]):
        self.X = X
        self.y = y
        self.time_vecs = time_vecs
        self.n_iter = n_iter
        self.burn = burn
        self.n_dept = len(self.X)
        self.n_par = self.X[0].shape[1]
        self.number_obs_in_each_dept = np.array([X_dept.shape[0] for X_dept in self.X])
        self.betas, self.sigmas_diag, self.mu, self.bandwidths, self.sigmas_squared, self.tau_sq_1s, self.f, self.gp_mats = self._initialize_parameters(
            betas_start, sigmas_start, mu_start, sigmas_diag_start, bandwidth_start, tau_sq_1_start, f_start)
        self.accept = None
        self.traces = self._initialize_traces()
        self.sigmas_proposal = sigmas_proposal

    def _initialize_parameters(self, betas_start, sigmas_start, mu_start, sigmas_diag_start, bandwidth_start,
                               tau_sq_1_start, f_start):
        betas = np.ones([self.n_par, len(self.X)]) * betas_start
        sigmas_diag = np.ones(self.n_par) * sigmas_diag_start
        mu = np.ones(self.n_par) * mu_start
        bandwidths = np.ones(self.n_dept) * bandwidth_start
        sigmas_squared = np.ones(self.n_dept) * sigmas_start
        tau_sq_1s = np.ones(self.n_dept) * tau_sq_1_start
        f = [np.ones(self.number_obs_in_each_dept[n]) * f_start for n in range(self.n_dept)]
        gp_mats = [self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=bandwidths[dept],
                                                              tau_sq_1=tau_sq_1s[dept]) for dept in range(self.n_dept)]
        return betas, sigmas_diag, mu, bandwidths, sigmas_squared, tau_sq_1s, f, gp_mats

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
                  'accept': np.zeros((self.n_iter, self.n_dept))}
        return traces

    def _update_cov_mats(self):
        alpha = np.ones(self.n_dept) * self.number_obs_in_each_dept / 2
        beta = np.array([0.5 * (self.y[i] - self.f[i]).T @ (
                self.y[i] - self.f[i]) for i in
                         range(self.n_dept)])
        sigmas = 1 / gamma(a=alpha, scale=1 / beta).rvs()
        self.sigmas_squared = sigmas
        self.cov_mats = [sigmas[n] * np.eye(n_obs) for n, n_obs in enumerate(self.number_obs_in_each_dept)]

    def _update_mu(self):
        mean = np.mean(self.betas, axis=1)
        cov = 1 / self.n_dept * np.diag(self.sigmas_diag)
        self.mu = multivariate_normal(mean, cov, allow_singular=True).rvs()

    def _update_sigmas_diag(self):
        alpha = np.ones(self.n_par) * (self.n_dept + 1) / 2
        beta = 0.5 * (((self.betas.T - self.mu)**2).sum(axis=0) + 1)
        self.sigmas_diag = 1 / gamma(a=alpha, scale=1 / beta).rvs()

    def _update_f(self):
        for dept in range(self.n_dept):
            cov = np.linalg.inv(
                1 / self.sigmas_squared[dept] * np.eye(self.number_obs_in_each_dept[dept]) + np.linalg.inv(
                    self.gp_mats[dept]))
            m = cov @ (np.linalg.inv(self.gp_mats[dept]) @ self.X[dept] @ self.betas[:, dept] + 1 / self.sigmas_squared[
                dept] * self.y[dept])
            self.f[dept] = multivariate_normal(mean=m, cov=cov, allow_singular=True).rvs()

    def _update_tau_sq_1s_and_bandwidth(self):
        self.accept = np.zeros(self.n_dept)
        for dept in range(self.n_dept):
            new_tau_sq_1 = np.max([.5, norm(loc=self.tau_sq_1s[dept], scale=self.sigmas_proposal[0]).rvs()])
            new_bandwidth = np.max([.5, norm(loc=self.bandwidths[dept], scale=self.sigmas_proposal[1]).rvs()])
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=new_bandwidth,
                                                              tau_sq_1=new_tau_sq_1)

            new_log_prior_tau = uniform.logpdf(new_tau_sq_1, loc=1, scale=99)
            new_log_prior_b = uniform.logpdf(new_bandwidth, loc=1, scale=149)
            new_log_likelihood = self._draw_log_likelihood(dept, new_cov)
            new_log_posterior = new_log_likelihood + new_log_prior_b + new_log_prior_tau

            old_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=self.tau_sq_1s[dept])
            old_log_prior_tau = uniform.logpdf(self.tau_sq_1s[dept], loc=1, scale=99)
            old_log_prior_b = uniform.logpdf(self.bandwidths[dept], loc=1, scale=149)
            old_log_likelihood = self._draw_log_likelihood(dept, old_cov)
            old_log_posterior = old_log_likelihood + old_log_prior_b + old_log_prior_tau
            log_ratio = new_log_posterior - old_log_posterior
            accept = self._accept_or_reject(log_ratio)
            semi_definite = np.all(np.linalg.eigvals(new_cov) > 0)
            if accept and semi_definite:
                self.tau_sq_1s[dept] = new_tau_sq_1
                self.bandwidths[dept] = new_bandwidth
                self.gp_mats[dept] = new_cov
                self.accept[dept] = 1



    @staticmethod
    def _accept_or_reject(log_ratio):
        r = uniform.rvs()
        if log_ratio > np.log(r):
            return True
        else:
            return False

    def _draw_log_likelihood(self, dept, cov):
        log_likelihood = multivariate_normal(mean=self.X[dept] @ self.betas[:, dept], cov=cov, allow_singular=True).logpdf(
            self.f[dept])
        return log_likelihood

    @staticmethod
    def _calculate_exp_covariance_function(x_1, x_2, bandwidth, tau_sq_1, tau_sq_2=1e-5):
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
                self._update_tau_sq_1s_and_bandwidth()
                self._update_f()
                self._update_cov_mats()
                self._update_sigmas_diag()
                self._update_mu()
                self._update_betas()
                self._update_traces(it)
            except:
                print('pb')

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
        self.traces['accept'][it, :] = self.accept

        for n in range(self.n_dept):
            self.traces['f'][n][:, it] = self.f[n]

import numpy as np
from scipy.stats import multivariate_normal, gamma, uniform, norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt



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
        self.bandwidths = np.ones(self.n_dept) * 40
        self.tau_sq_1s = np.ones(self.n_dept) * 50
        self.f = [np.zeros(self.number_obs_in_each_dept[n]) for n in range(self.n_dept)]
        self.gp_mats = [np.eye(self.number_obs_in_each_dept[dept]) for dept in
                        range(self.n_dept)]

        self.bandwidths_posteriors = np.array(
            [self._calculate_bandwidth_posterior(dept, self.bandwidths[dept], self.gp_mats[dept]) for dept in
             range(self.n_dept)])
        self.tau_sq_posteriors = np.array(
            [self._calculate_tau_sq_1_posterior(dept, self.tau_sq_1s[dept], self.gp_mats[dept]) for dept in
             range(self.n_dept)])

        self.accept_tau = np.zeros([self.n_iter, self.n_dept])
        self.accept_bandwidth = np.zeros([self.n_iter, self.n_dept])

        self.sigmas_squared = np.ones(self.n_dept)


        self.traces = {'betas': np.ones([self.n_iter, self.n_par, self.n_dept]) * np.nan,
                       'mu': np.ones((self.n_iter, self.n_par)) * np.nan,
                       'taus': np.ones((self.n_iter, self.n_par)) * np.nan,
                       'bandwidths': np.ones((self.n_iter, self.n_dept)) * np.nan,
                       'sigmas_squared': np.ones((self.n_iter, self.n_dept)) * np.nan,
                       'tau_sq_1s': np.ones((self.n_iter, self.n_dept)) * np.nan,
                       'f': [
                           np.ones([self.number_obs_in_each_dept[n], self.n_iter]) * np.nan for
                           n in range(self.n_dept)]}
        self.burn = burn

    def _update_cov_mats(self):
        alpha = (self.number_obs_in_each_dept + 1) / 2
        beta = np.array([0.5 * (self.y[i] - (self.X[i] @ self.betas[:, i])).T @ (
                    self.y[i] - (self.X[i] @ self.betas[:, i] )) for i in
                         range(self.n_dept)])
        sigmas = 1 / gamma(a=alpha, scale=1 / beta).rvs()
        self.sigmas_squared = sigmas
        self.cov_mats = [sigmas[n] * np.eye(n_obs) for n, n_obs in enumerate(self.number_obs_in_each_dept)]

    def _update_mu(self):
        mean = 1 / self.n_dept * np.mean(self.betas, axis=1)
        cov = 1 / self.n_dept * np.diag(self.taus)
        self.mu = multivariate_normal(mean, cov, allow_singular=True).rvs()

    def _update_taus(self):
        alpha = (self.n_dept + 1) / 2
        beta = 0.5 * (np.sum((self.betas - self.mu[:, None]) ** 2, axis=1) + 1)
        self.taus = 1 / gamma(a=alpha, scale=1 / beta).rvs()

    def _update_f(self):
        for dept in range(self.n_dept):
            cov = np.linalg.inv(
                1 / self.sigmas_squared[dept] * np.eye(self.number_obs_in_each_dept[dept]) + np.linalg.inv(
                    self.gp_mats[dept]))
            m = 1 / self.sigmas_squared[dept] * cov @ (-self.X[dept] @ self.betas[:, dept] + self.y[dept])
            self.f[dept] = multivariate_normal(mean=m, cov=cov, allow_singular=True).rvs()

    def _update_tau_sq_1s_and_corr_mat(self, it):
        for dept in range(self.n_dept):
            new_log_tau_sq = norm(loc=np.log(self.tau_sq_1s[dept]), scale=.1).rvs()
            new_tau_sq = np.exp(new_log_tau_sq)
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=new_tau_sq)
            new_posterior = self._calculate_tau_sq_1_posterior(dept, new_tau_sq, new_cov)
            old_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=self.tau_sq_1s[dept])
            old_posterior = self._calculate_tau_sq_1_posterior(dept, self.tau_sq_1s[dept], old_cov)
            log_ratio = self._calculate_log_metropolis_hastings_ratio(new_posterior, old_posterior)
            accept = self._accept_or_reject(log_ratio)
            self.accept_tau[it, dept] = accept
            if accept:
                self.tau_sq_1s[dept] = new_tau_sq
                self.tau_sq_posteriors[dept] = new_posterior
                self.gp_mats[dept] = new_cov

    def _update_bandwidths(self, it):
        for dept in range(self.n_dept):
            new_log_bandwidth = norm(loc=np.log(self.bandwidths[dept]), scale=.1).rvs()
            new_bandwidth = np.exp(new_log_bandwidth)
            new_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=new_bandwidth,
                                                              tau_sq_1=self.tau_sq_1s[dept])
            new_posterior = self._calculate_bandwidth_posterior(dept, new_log_bandwidth, new_cov)

            old_cov = self._calculate_exp_covariance_function(x_1=self.time_vecs[dept], x_2=self.time_vecs[dept],
                                                              bandwidth=self.bandwidths[dept],
                                                              tau_sq_1=self.tau_sq_1s[dept])
            old_posterior = self._calculate_bandwidth_posterior(dept, self.bandwidths[dept], old_cov)
            log_ratio = self._calculate_log_metropolis_hastings_ratio(new_posterior, old_posterior)
            accept = self._accept_or_reject(log_ratio)
            self.accept_bandwidth[it, dept] = accept

            if 1 == 0:

                rg = np.linspace(self.bandwidths[dept] - 15, self.bandwidths[dept] + 100, 100)
                covs = [self._calculate_exp_covariance_function(x_1=self.time_vectors[dept], x_2=self.time_vectors[dept],
                                                                bandwidth=val,
                                                                tau_sq_1=self.tau_sq_1s[dept]) for val in rg]
                posts = [self._calculate_bandwidth_posterior(dept, rg[n], covs[n]) for n in range(100)]
                plt.plot(rg, posts)

                plt.axvline(x=self.bandwidths[dept], label='old', c='blue')

                plt.axvline(x=new_bandwidth, label='new', c='black')
                plt.legend()
                plt.title(dept)
                plt.xlabel(accept)
                plt.show()
                plt.figure()
            if accept:

                self.bandwidths[dept] = new_bandwidth
                self.bandwidths_posteriors[dept] = new_posterior

    def _calculate_bandwidth_posterior(self, dept, bandwidth, cov):
        likelihood = self._draw_likelihood(self.f[dept], cov)
        prior = uniform.pdf(bandwidth, loc=1, scale=1000)
        return likelihood * prior

    def _calculate_tau_sq_1_posterior(self, dept, tau_sq_1, cov):
        likelihood = self._draw_likelihood(self.f[dept], cov)
        prior = uniform.pdf(tau_sq_1, loc=1, scale=10000)
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
    def _draw_likelihood(f, cov):
        likelihood = multivariate_normal(mean=[0] * len(f), cov=cov).pdf(f)
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
            self.betas[:, dept_idx] = multivariate_normal(mean, cov, allow_singular=True).rvs()

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            try:
                self._update_bandwidths(it)
                self._update_tau_sq_1s_and_corr_mat(it)
                self._update_f()
                self._update_cov_mats()
                self._update_taus()
                self._update_mu()
                self._update_betas()
                self._update_traces(it)
            except:
              # # warnings.warn("problem")
                print('PROBLEM')

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
        self.traces['taus'][it, :] = self.taus
        self.traces['bandwidths'][it, :] = self.bandwidths
        self.traces['tau_sq_1s'][it, :] = self.tau_sq_1s
        for n in range(self.n_dept):
            self.traces['f'][n][:, it] = self.f[n]

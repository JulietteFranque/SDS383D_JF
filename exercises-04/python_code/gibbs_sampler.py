import numpy as np
from scipy.stats import multivariate_normal, gamma, norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class Initializer:
    def __init__(self, df, theta=None, mu=None, sigma_squared=None, tau_squared=None):
        self.df = df
        self.P = self._calculate_number_of_groups()
        self.n_i = self._calculate_number_of_people_per_group()
        self.mean_per_group = self._calculate_mean_per_group()
        self.theta = self._initialize_theta(theta)
        self.mu = self._initialize_mu(mu)
        self.sigma_squared = self._initialize_sigma_squared(sigma_squared)
        self.tau_squared = self._initialize_tau_squared(tau_squared)
        self.total_people = self.n_i.sum()

    def _calculate_mean_per_group(self):
        return self.df.groupby('group').mean().to_numpy().flatten()

    def _initialize_theta(self, theta):
        if theta is not None:
            return theta
        return self.mean_per_group

    @staticmethod
    def _initialize_mu(mu):
        if mu is not None:
            return mu
        return 0

    @staticmethod
    def _initialize_sigma_squared(sigma_squared):
        if sigma_squared is not None:
            return sigma_squared
        return 1

    @staticmethod
    def _initialize_tau_squared(tau_squared):
        if tau_squared is not None:
            return tau_squared
        return 1

    def _calculate_number_of_groups(self):
        P = self.df.groupby('group').ngroups
        return P

    def _calculate_number_of_people_per_group(self):
        n_i = self.df.groupby('group').size().to_numpy()
        return n_i


class GibbsSampler(Initializer):
    def __init__(self, df, theta=None, mu=None, sigma_squared=None, tau_squared=None, n_iter=5000, burn=100):
        super().__init__(df=df, theta=theta, mu=mu, sigma_squared=sigma_squared, tau_squared=tau_squared)
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {'sigma_squared': np.zeros(self.n_iter),
                       'tau_squared': np.zeros(self.n_iter),
                       'mu': np.zeros(self.n_iter),
                       'theta': np.zeros((self.n_iter, self.P))}

    def _update_tau_squared(self, dist_mu):
        alpha = (self.P + 1) / 2
        beta = 1 / 2 * ((self.theta - dist_mu).T @ np.linalg.inv(self.sigma_squared * np.identity(self.P)) @ (
                self.theta - dist_mu) + 1)
        self.tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_sigma_squared(self, dist_mu):
        alpha = (self.total_people + self.P) / 2
        beta_first_term = (self.theta - dist_mu).T @ np.linalg.inv(self.tau_squared * np.identity(self.P)) @ (
                self.theta - dist_mu)
        beta_second_term = ((self.theta[self.df['group'].values - 1] - self.df['values']) ** 2).sum()
        beta = 0.5 * (beta_first_term + beta_second_term)
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_mu(self):
        mean = self.theta.mean()
        var = self.sigma_squared * self.tau_squared / self.P
        self.mu = norm.rvs(loc=mean, scale=np.sqrt(var))

    def _update_theta(self, dist_mu):
        mean = (self.mean_per_group * self.tau_squared * self.n_i + dist_mu) / (self.tau_squared * self.n_i + 1)
        cov_matrix = self.sigma_squared * self.tau_squared / (self.tau_squared * self.n_i + 1) * np.identity(self.P)
        self.theta = multivariate_normal.rvs(mean=mean, cov=cov_matrix)

    def _update_traces(self, it):
        self.traces['theta'][it, :] = self.theta
        self.traces['mu'][it] = self.mu
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared

    def _remove_burn(self):
        for trace in self.traces.keys():
            self.traces[trace] = self.traces[trace][self.burn:]

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_mu()
            self._update_sigma_squared(self.mu)
            self._update_tau_squared(self.mu)
            self._update_theta(self.mu)
            self._update_traces(it)

        self._remove_burn()

    def plot_theta_histograms(self):
        traces = self.traces['theta']
        for group in range(self.P):
            plt.hist(traces[:, group], density=True, alpha=.5, bins=50)
        plt.title('theta posteriors')
        plt.figure()

    def plot_other_histograms(self, variable):
        trace = self.traces[variable]
        plt.hist(trace, density=True, alpha=.5, bins=50)
        plt.title(f'{variable} posterior')
        plt.figure()

    def plot_all_posteriors(self):
        self.plot_theta_histograms()
        keys = list(self.traces.keys())
        keys.remove('theta')
        for trace in keys:
            self.plot_other_histograms(trace)

## stuff below gives weird results

class GibbsSamplerBeta(GibbsSampler):
    def __init__(self, df, theta=None, mu=None, sigma_squared=None, tau_squared=None, n_iter=5000, burn=100, beta=None, group_controls = np.array([0, 1])):
        super().__init__(df=df, theta=theta, mu=mu, sigma_squared=sigma_squared, tau_squared=tau_squared, n_iter=n_iter, burn=burn)
        self.beta = self._initialize_beta(beta)
        self.traces = {'sigma_squared': np.zeros(self.n_iter),
                       'tau_squared': np.zeros(self.n_iter),
                       'mu': np.zeros(self.n_iter),
                       'theta': np.zeros((self.n_iter, self.P)),
                       'beta': np.zeros(self.n_iter)}
        self.group_controls = group_controls

    @staticmethod
    def _initialize_beta(beta):
        if beta is not None:
            return beta
        return 5

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            self._update_mu()
            self._update_beta()
            dist_mu = self.mu - self.beta * self.group_controls
            self._update_sigma_squared(dist_mu)
            self._update_tau_squared(dist_mu)
            self._update_theta(dist_mu)
            self._update_traces(it)

    def _update_beta(self):
        mean = (np.mean(self.theta * self.group_controls) - self.mu * np.mean(
            self.group_controls)) / np.mean(self.group_controls ** 2)
        var = self.sigma_squared * self.tau_squared / (np.mean(self.group_controls ** 2) * self.P)
        self.beta = norm(loc=mean, scale=np.sqrt(var)).rvs()

    def _update_traces(self, it):
        self.traces['theta'][it, :] = self.theta
        self.traces['mu'][it] = self.mu
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared
        self.traces['beta'][it] = self.beta

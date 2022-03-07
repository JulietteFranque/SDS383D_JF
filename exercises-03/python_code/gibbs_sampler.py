from scipy.stats import multivariate_normal, gamma, multivariate_t
import numpy as np
from scipy import linalg
from tqdm import tqdm
from python_code.homoskedastic_model import HomoskedasticModel as Model
import matplotlib.pyplot as plt


class GibbsSampler(Model):
    def __init__(self, X, y, lam=None, K=None, m=None, d=None, eta=None, fit_intercept=True, omega=None, h=None,
                 coefs=None, n_iter=1000, n_to_discard=100):
        super().__init__(X, y, lam=lam, K=K, m=m, d=d, eta=eta, fit_intercept=fit_intercept)
        self.omega = self._initialize_omega(omega)
        self.h = self._initialize_h(h)
        self.coefs = self._initialize_coefs(coefs)
        self.n_iter = n_iter
        self.traces = {

            'beta_trace': np.zeros([self.n_iter, self.X.shape[1]]),
            'Lambda_trace': np.zeros([self.n_iter, self.X.shape[0]]),
            'omega_trace': np.zeros(self.n_iter),

        }
        self.n_to_discard = n_to_discard
        self.traces_kept = None

    @staticmethod
    def _initialize_omega(omega):
        if omega is not None:
            return omega
        return 1

    @staticmethod
    def _initialize_h(h):
        if h is not None:
            return h
        return 1

    def _initialize_coefs(self, coefs):
        if coefs is not None:
            return coefs
        return np.ones_like(self.X.shape[1])

    def _update_lambda(self):
        alpha = (self.h + 1) / 2
        beta = 1 / (self.h + self.omega * (self.y - self.X @ self.coefs)) / 2
        samples = gamma.rvs(alpha, beta)
        self.lam = np.diag(samples)

    def _update_coefs(self, m_star, precision_matrix):
        self.coefs = multivariate_normal(mean=m_star, cov=linalg.inv(precision_matrix)).rvs()

    def _update_omega(self, d_star, eta_star):
        self.omega = gamma.rvs(d_star / 2, 2 / eta_star)

    def _update_traces(self, it):
        self.traces['Lambda_trace'][it, :] = np.diag(self.lam)
        self.traces['beta_trace'][it, :] = self.coefs
        self.traces['omega_trace'][it] = self.omega

    def fit(self):
        d_star = self._calculate_d_star()
        for it in tqdm(range(self.n_iter)):
            eta_star = self._calculate_eta_star()
            K_star = self._calculate_K_star()
            m_star = self._calculate_m_star()
            precision_matrix = self.omega * K_star
            try:
                self._update_coefs(m_star, precision_matrix)
                self._update_omega(d_star, eta_star)
                self._update_lambda()
                self._update_traces(it)
            except ValueError:
                pass
        self.traces_kept = self._discard_burn_in_steps()

    def _discard_burn_in_steps(self):
        traces_to_keep = {

            'beta_trace': np.zeros([self.n_iter - self.n_to_discard, self.X.shape[1]]),
            'Lambda_trace': np.zeros([self.n_iter - self.n_to_discard, self.X.shape[0]]),
            'omega_trace': np.zeros(self.n_iter - self.n_to_discard),

        }
        for key in self.traces.keys():
            traces_to_keep[key] = self.traces[key][self.n_to_discard:]
        return traces_to_keep

    def _calculate_averaged_coefficients(self):
        averaged_coefs = np.mean(self.traces_kept['beta_trace'], axis=0)
        return averaged_coefs

    def predict(self, X):
        X = self._initialize_X(X)
        coefs = self._calculate_averaged_coefficients()
        return X @ coefs

    def calculate_confidence_intervals(self, alpha=0.05):
        traces = self.traces_kept['beta_trace']
        lower = np.quantile(traces, alpha / 2, axis=0)
        higher = np.quantile(traces, 1 - alpha / 2, axis=0)
        return lower, higher

    def plot_coefs_histograms(self):
        fig, ax = plt.subplots(figsize=(15, 9))
        traces = self.traces_kept['beta_trace']
        num_coefs = len(self.coefs)
        for i in range(num_coefs):
            plt.subplot(int(num_coefs / 3) + 1 * (num_coefs % 3 != 0), 3, i + 1)
            plt.hist(traces[:, i], density=True, alpha=.5, bins=50)
            plt.title(f'coef {i}')
        return ax, fig

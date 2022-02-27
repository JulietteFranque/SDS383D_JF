from scipy.stats import multivariate_normal, gamma, multivariate_t
import numpy as np
from scipy import linalg
from tqdm import tqdm
from python_code.homoskedastic_model import HomoskedasticModel as Model


class GibbsSampler(Model):
    def __init__(self, X, y, lam=None, K=None, m=None, d=None, eta=None, fit_intercept=True, omega=None, h=None,
                 betas=None, n_iter=1000):
        super().__init__(X, y, lam=lam, K=K, m=m, d=d, eta=eta, fit_intercept=fit_intercept)
        self.omega = self._initialize_omega(omega)
        self.h = self._initialize_h(h)
        self.betas = self._initialize_betas(betas)
        self.n_iter = n_iter
        self.traces = {

            'beta_trace': np.zeros([self.n_iter, self.X.shape[1]]),
            'Lambda_trace': np.zeros([self.n_iter, self.X.shape[0]]),
            'omega_trace': np.zeros(self.n_iter),

        }

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

    def _initialize_betas(self, betas):
        if betas is not None:
            return betas
        return np.ones_like(self.X.shape[1])

    def _update_lambda(self):
        alpha = (self.h + 1) / 2
        beta = 1 / (self.h + self.omega * (self.y - self.X @ self.betas)) / 2
        samples = gamma.rvs(alpha, beta)
        self.lam = np.diag(samples)

    def _update_betas(self, m_star, precision_matrix):
        self.betas = multivariate_normal(mean=m_star, cov=linalg.inv(precision_matrix)).rvs()

    def _update_omega(self, d_star, eta_star):
        self.omega = gamma.rvs(d_star / 2, 2 / eta_star)

    def _update_traces(self, it):
        self.traces['Lambda_trace'][it, :] = np.diag(self.lam)
        self.traces['beta_trace'][it, :] = self.betas
        self.traces['omega_trace'][it] = self.omega

    def fit(self):
        for it in tqdm(range(self.n_iter)):
            d_star = self._calculate_d_star()
            eta_star = self._calculate_eta_star()
            K_star = self._calculate_K_star()
            m_star = self._calculate_m_star()
            precision_matrix = self.omega * K_star
            try:
                self._update_betas(m_star, precision_matrix)
                self._update_omega(d_star, eta_star)
                self._update_lambda()
                self._update_traces(it)
            except ValueError:
                pass

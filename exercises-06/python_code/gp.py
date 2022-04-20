from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal, norm
import numpy as np
from scipy.optimize import minimize

class GaussianProcess:
    def __init__(self, x_data, y_data, tau_sq_1=None, sigma_squared=1, tau_sq_2=1e-6, bandwidth=None):
        self.x = x_data
        self.y = y_data
        self.tau_sq_1 = tau_sq_1
        self.tau_sq_2 = tau_sq_2
        self.bandwidth = bandwidth
        self.sigma_squared = sigma_squared

    def _calculate_exp_covariance_function(self, x_1, x_2):
        distance = distance_matrix(x_1, x_2)
        cov = self.tau_sq_1 * np.exp(-1 / 2 * (distance / self.bandwidth) ** 2) + self.tau_sq_2 * np.eye(x_1.shape[0], x_2.shape[0])
        return cov

    def _calculate_marginal_P_y(self, parameters):
        self.bandwidth, self.tau_sq_1 = parameters
        C = self._calculate_exp_covariance_function(self.x, self.x)
        cov = self.sigma_squared * np.eye(self.x.shape[0]) + C
        dist = multivariate_normal(cov=cov, allow_singular=True)
        return dist.logpdf(self.y)

    def _calculate_GP_parameters(self, x_pred):
        n_points = len(self.x)
        C11 = self._calculate_exp_covariance_function(x_1=self.x, x_2=self.x)
        C22 = self._calculate_exp_covariance_function(x_1=x_pred, x_2=x_pred)
        C21 = self._calculate_exp_covariance_function(x_1=self.x, x_2=x_pred)
        inv_mat = np.linalg.pinv(C11 + np.eye(n_points) * self.sigma_squared)
        H = C21.T @ inv_mat
        mean = H @ self.y
        cov = C22 - (C21.T @ inv_mat @ C21)
        return H, mean, cov

    def draw_GP_curves(self, x_pred, n_pred=10):
        _, mean, cov = self._calculate_GP_parameters(x_pred)
        dist = multivariate_normal(mean, cov)
        return dist.rvs(n_preed)

    def _find_optimal_parameters(self):
        minimize(lambda parameters: -self._calculate_marginal_P_y(parameters), x0=np.array([1, 1]), bounds=((0, None),(0, None)), method='Powell')

    def fit(self):
        self. _find_optimal_parameters()

    def predict(self, x_pred):
        H, y_pred, _ = self._calculate_GP_parameters(x_pred)
        CI = self._calculate_CI(y_pred, H)
        return y_pred, CI

    def _calculate_CI(self, y_pred, H, sig_level=.05):
        var = self.sigma_squared * np.sum(H ** 2, axis=1)
        z = norm(0, 1).ppf(1 - sig_level / 2)
        lower = y_pred.flatten() - z * np.sqrt(var)
        upper = y_pred.flatten() + z * np.sqrt(var)
        return lower, upper




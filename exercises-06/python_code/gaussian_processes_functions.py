from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
import numpy as np


def calculate_matern_covariance_function(x, b, tau_sq_1, tau_sq_2):
    distance = distance_matrix(x, x)
    C = tau_sq_1 * np.exp(-1 / 2 * (distance / b) ** 2) + tau_sq_2 * np.eye(x.shape[0])
    return C


def calculate_matern_covariance_function_52(x, b, tau_sq_1, tau_sq_2):
    distance = distance_matrix(x, x)
    C = tau_sq_1 * (1 + np.sqrt(5) * distance / b + 5 * distance ** 2 / (3 * b ** 2)) * np.exp(
        -np.sqrt(5) * distance / b) + tau_sq_2 * np.eye(x.shape[0])
    return C


def draw_GP(x_data, y_data, x_pred, sigma_squared, tau_1=.5, tau_2=1e-6, b=.5, n_predictions=1):
    mean, cov = get_GP_parameters(x_data, y_data, x_pred, tau_1, tau_2, sigma_squared, b)
    dist = multivariate_normal(mean, cov)
    return dist.rvs(size=n_predictions)


def get_GP_parameters(x_data, y_data, x_pred, tau_1, tau_2, sigma_squared, b):
    n_points = len(x_data)
    cov_mat = calculate_matern_covariance_function(np.concatenate([x_data.reshape(-1, 1), x_pred.reshape(-1, 1)]), b,
                                                   tau_1, tau_2)
    sigma_11 = cov_mat[:len(x_data), :len(x_data)]
    sigma_22 = cov_mat[len(x_data):, len(x_data):]
    sigma_21 = cov_mat[len(x_data):, :len(x_data)]
    sigma_12 = sigma_21.T
    mean = sigma_21 @ np.linalg.inv(sigma_11) @ y_data
    cov = sigma_22 - (sigma_21 @ np.linalg.inv(sigma_11 + sigma_squared * np.eye(n_points)) @ sigma_12)
    return mean, cov

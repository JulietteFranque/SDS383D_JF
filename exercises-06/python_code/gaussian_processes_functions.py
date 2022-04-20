from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
import numpy as np
from scipy.optimize import minimize


def calculate_k_matrix(x_1, x_2):
    return (x_1.flatten()[:, None] == x_2.flatten()[None, :]).astype(int)


def calculate_matern_covariance_function(x_1, x_2, bandwidth, tau_sq_1, tau_sq_2):
    distance = distance_matrix(x_1, x_2)
    k_delta_matrix = calculate_k_matrix(x_1, x_2)
    C = tau_sq_1 * np.exp(-1 / 2 * (distance / bandwidth) ** 2) + tau_sq_2 * k_delta_matrix
    return C

def calculate_matern_covariance_function_52(x_1, x_2, bandwidth, tau_sq_1, tau_sq_2):
    distance = distance_matrix(x_1, x_2)
    k_delta_matrix = calculate_k_matrix(x_1, x_2)
    C = tau_sq_1 * (1 + np.sqrt(5) * distance / bandwidth + 5 * distance ** 2 / (3 * bandwidth ** 2)) * np.exp(
        -np.sqrt(5) * distance / bandwidth) + tau_sq_2 * k_delta_matrix
    return C


def draw_GP_smoothing(x_data, y_data, x_pred, sigma_squared, tau_sq_1, tau_sq_2, bandwidth, n_predictions=1):
    mean, cov = get_GP_parameters(x_data, y_data, x_pred, tau_sq_1, tau_sq_2, sigma_squared, bandwidth)
    dist = multivariate_normal(mean, cov, allow_singular=True)
    return dist.rvs(size=n_predictions)


def get_GP_parameters(x_data, y_data, x_pred, tau_sq_1, tau_sq_2, sigma_squared, bandwidth):
    n_points = len(x_data)
    sigma_11 = calculate_matern_covariance_function(x_1=x_data, x_2=x_data, bandwidth=bandwidth, tau_sq_1=tau_sq_1, tau_sq_2=tau_sq_2)
    sigma_22 = calculate_matern_covariance_function(x_1=x_pred, x_2=x_pred, bandwidth=bandwidth, tau_sq_1=tau_sq_1, tau_sq_2=tau_sq_2)
    sigma_21 = calculate_matern_covariance_function(x_1=x_data, x_2=x_pred, bandwidth=bandwidth, tau_sq_1=tau_sq_1, tau_sq_2=tau_sq_2)

    pinv_mat = np.linalg.pinv(sigma_11 + np.eye(n_points)*sigma_squared)
    mean = sigma_21.T @ pinv_mat @ y_data
    cov = sigma_22 - (sigma_21 @ pinv_mat @ sigma_21.T)
    return mean, cov

def calculate_GP_posterior(x, y, sigma_squared, bandwidth, tau_sq_1, tau_sq_2):
    C = calculate_matern_covariance_function(x_1=x.reshape(-1, 1), x_2=x.reshape(-1, 1), bandwidth=bandwidth, tau_sq_1=tau_sq_1, tau_sq_2=tau_sq_2)
    cov_mat = np.linalg.pinv(np.eye(x.shape[0]) / sigma_squared + np.linalg.pinv(C))
    mean = 1 / sigma_squared * cov_mat @ y
    return mean, cov_mat

def draw_from_posterior(x, y, sigma_squared, bandwidth, tau_sq_1, tau_sq_2, n_points=100):
    mean, cov_mat = calculate_GP_posterior(x, y, sigma_squared, bandwidth, tau_sq_1, tau_sq_2)
    dist = multivariate_normal(mean, cov_mat, allow_singular=True)
    return dist.rvs(size=n_points)

def calculate_CI(x, y, sigma_squared, bandwidth, tau_sq_1, tau_sq_2, n_points=1000):
    draws = draw_from_posterior(x, y, sigma_squared, bandwidth, tau_sq_1, tau_sq_2, n_points)
    CI = np.quantile(draws, [0.025, 0.975], axis=0)
    return CI

def calculate_marginal_P(x, y, parameters, sigma_squared, tau_sq_2, cov_function='exp'):
    b, tau_sq_1 = parameters
    if cov_function == 'exp':
        cov_fun = calculate_matern_covariance_function(x, x,  b, tau_sq_1, tau_sq_2)
    else:
        cov_fun = calculate_matern_covariance_function_52(x, x, b, tau_sq_1, tau_sq_2)
    cov_norm = sigma_squared * np.eye(x.shape[0]) + cov_fun
    dist = multivariate_normal(cov=cov_norm, allow_singular=True)
    return dist.logpdf(y)

def find_best_parameters(x, y, sigma_squared, tau_sq_2, cov_function='exp'):
    opt = minimize(
        lambda parameters: -calculate_marginal_P(x=x, y=y, parameters=parameters, sigma_squared=sigma_squared,
                                                 tau_sq_2=tau_sq_2, cov_function=cov_function), x0=np.array([1, 1]),
        method='Powell')
    return {'b': opt.x[0], 'tau': opt.x[1]}

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split


class KernelSmoother:
    def __init__(self, x, y, bandwidth=None):
        """can either fit bandwidth or specify bandwidth"""
        self.bandwidth = bandwidth
        self.x = x
        self.y = y

    def _get_weights(self, dist):
        x_kernel = dist / self.bandwidth
        kernel = 1 / self.bandwidth * 1 / (2 * np.pi) * np.exp(-x_kernel ** 2 / 2)
        weights = kernel / np.sum(kernel, axis=1)[:, np.newaxis]
        return weights

    def predict(self, x_predict):
        return self._get_y_smoothed(self.x, self.y, x_predict)

    def _get_y_smoothed(self, x_data, y_data, x_predict):
        """
        :param bandwidth: scalar
        :param x_data: (n_points x n_dim)
        :param y_data: (n_points x 1)
        :param x_predict: (n_points x n_dim)
        :return: smoothed points (n_points x 1)
        """
        dist = distance_matrix(x_predict, x_data)
        weights = self._get_weights(dist)
        y_smoothed = weights @ y_data
        return y_smoothed

    def _objective_function_basic_validation(self, bandwidth, x_train, x_test, y_train, y_test):
        self.bandwidth = bandwidth
        y_smoothed = self._get_y_smoothed(x_train, y_train, x_test)
        return mean_squared_error(y_smoothed, y_test)

    def _objective_function_leave_one_out_trick(self, bandwidth, x, y):
        self.bandwidth = bandwidth
        dist = distance_matrix(x, x)
        H_matrix = self._get_weights(dist)
        y_hat = self._get_y_smoothed(x, y, x).flatten()
        loocv = ((y.flatten() - y_hat).flatten() / (1 - np.diag(H_matrix))).T @ (
                    (y.flatten() - y_hat) / (1 - np.diag(H_matrix)))
        return loocv

    def fit(self, method='basic_validation'):
        """fit optimal bandwidth"""
        if method == 'leave_one_out':
            self.bandwidth = minimize(lambda b: self._objective_function_leave_one_out_trick(b, self.x, self.y), x0=1,
                                      bounds=[(0.05, None)]).x

        elif method == 'basic_validation':
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25)
            self.bandwidth = minimize(
                lambda b: self._objective_function_basic_validation(b, x_train, x_test, y_train, y_test), x0=1,
                bounds=[(0.05, None)]).x

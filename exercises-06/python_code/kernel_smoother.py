import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import mean_squared_error

class KernelSmoother:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def _get_weights(self, dist):
        x_kernel = dist / self.bandwidth
        kernel = 1 / self.bandwidth * 1 / (2 * np.pi) ** 0.5 * np.exp(-x_kernel ** 2 / 2)
        weights = kernel / np.sum(kernel, axis=1)[:, np.newaxis]
        return weights

    def predict(self, x_data, y_data, x_predict):
        """

        :param bandwidth: scalar
        :param x_data: (n_points x n_dim)
        :param y_data: (n_points x 1)
        :param x_predict: (n_points x n_dim)
        :return: smoothed points (n_points x 1)
        """
        dist = distance_matrix(x_predict, x_data)
        weights = self._get_weights(dist)
        y_smoothed = weights@y_data
        return y_smoothed

    def fit(self, x_train, x_test, y_train, y_test):
        bandwidths_to_try = np.linspace(0.3, 10, 500)
        mean_sq_errors = np.zeros(len(bandwidths_to_try))
        for n, bandwidth in enumerate(bandwidths_to_try):
            self.bandwidth = bandwidth
            y_smoothed = self.predict(x_train, y_train, x_test)
            mean_sq_errors[n] = mean_squared_error(y_smoothed, y_test)
        self.bandwidth = bandwidths_to_try[np.argmin(mean_sq_errors)]

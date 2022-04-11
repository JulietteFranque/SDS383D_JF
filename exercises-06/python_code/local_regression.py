import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


class Lowess:
    def __init__(self, degree=1):
        self.x_data = None
        self.y_data = None
        self.degree = degree
        self.bandwidth = None

    def _make_R_matrix(self, x_predict):
        R = np.tile(self.x_data - x_predict[:, :, None], self.degree + 1)
        for col_degree in range(self.degree + 1):
            R[:, :, col_degree] = R[:, :, col_degree] ** col_degree
        return R

    def _calculate_weights(self, dist):
        x_kernel = dist / self.bandwidth
        kernel = 1 / self.bandwidth * 1 / (2 * np.pi) * np.exp(-x_kernel ** 2 / 2)
        weights = kernel / np.sum(kernel, axis=1)[:, np.newaxis]
        return weights

    def _get_distances(self, x_predict):
        dist = distance_matrix(x_predict, self.x_data)
        return dist

    def _calculate_smoothing_matrix(self, x_predict):
        R = self._make_R_matrix(x_predict)
        dist = self._get_distances(x_predict)
        weights = self._calculate_weights(dist)
        R_transpose = np.transpose(R, axes=[0, 2, 1])
        H = np.linalg.solve(R_transpose * weights[:, None, :] @ R, R_transpose * weights[:, None, :])
        H_at_target = H[:, 0, :]
        #H_at_target_normalized = H_at_target / np.sum(H_at_target, axis=1)[:, np.newaxis]
        # for each target point, the estimated point is the first element only
        return H_at_target

    def predict(self, x_predict):
        H = self._calculate_smoothing_matrix(x_predict)
        y_pred = (H @ self.y_data)
        return y_pred

    def _objective_function_leave_one_out(self, bandwidth):
        self.bandwidth = bandwidth
        H_matrix = self._calculate_smoothing_matrix(self.x_data)
        y_hat = self.predict(self.x_data).flatten()
        loocv = ((self.y_data.flatten() - y_hat).flatten() / (1 - np.diag(H_matrix))).T @ (
                (self.y_data.flatten() - y_hat) / (1 - np.diag(H_matrix)))
        return loocv

    def fit(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.bandwidth = minimize(self._objective_function_leave_one_out, x0=np.array([1]),
                                  bounds=[(0.05, None)]).x

import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, fit_intercept=False, scale=True, max_iteration=8000, tol=1e-6, learning_rate=1):
        """
        Fit logistic regression

        Parameters
        ----------
        fit_intercept: bool
            fit intercept or not
        max_iteration: int
            maximum number of iterations
        tol: float
            tolerance criteria for stopping gradient descent
        learning_rate: int
            learning rate for gradient descent

        """
        self.betas = None
        self.scale = scale
        self.standard_error = None
        self.log_likelihood = None
        self.fit_intercept = fit_intercept
        self.max_iteration = max_iteration
        self.tol = tol
        self.learning_rate = learning_rate
        self.log_likelihood = np.array([None] * self.max_iteration)
        self.last_iteration = None
        self.methods = {'Gradient descent': self._do_gradient_descent, 'Newton': self._do_newton}

    @staticmethod
    def _scale_data(X):
        """scale data w mean & std"""
        return (X - X.mean(axis=0))/X.std(axis=0)

    def fit(self, X, y, method='Gradient descent'):
        """train model"""
        if self.scale:
            X = self._scale_data(X)
        if self.fit_intercept:
            X = self._make_intercept(X)
        training_result, it_number = self.methods[method](X, y)
        self.standard_error = self.calculate_standard_error(self.betas, X)
        self.log_likelihood = self.log_likelihood[0:it_number]
        self.last_iteration = it_number
        return training_result

    def calculate_standard_error(self, betas, X):
        hessian = self._calculate_hessian(X, betas)
        inv_hessian = np.linalg.inv(hessian)
        standard_var = -np.diag(inv_hessian)
        standard_error = np.sqrt(standard_var)
        return standard_error

    @staticmethod
    def _make_intercept(X):
        """add intercept"""
        column_ones = np.ones(X.shape[0])[:, None]
        bigger_X = np.hstack((column_ones, X))
        return bigger_X

    def predict(self, X):
        """add intercept"""
        if self.scale:
            X = self._scale_data(X)
        if self.fit_intercept:
            X = self._make_intercept(X)
        prediction = self._get_mu(X, self.betas)
        return prediction

    def _check_tolerance(self, iteration):
        try:
            log_likelihood_percent_change = np.abs(
                (self.log_likelihood[iteration] - self.log_likelihood[iteration - 1]) / self.log_likelihood[
                    iteration - 1])
            if log_likelihood_percent_change < self.tol:
                return True
        except TypeError:
            pass

    def _do_newton(self, X, y):
        """ use newton to fit model"""
        self.betas = np.random.uniform(size=X.shape[1])
        for it in tqdm(range(self.max_iteration)):
            hessian = self._calculate_hessian(X, self.betas)
            gradient = self._get_score_function(X, y, self.betas)
            delta_betas = np.linalg.solve(hessian, self.learning_rate * gradient)
            self.betas = self.betas + delta_betas
            log_likelihood = self._calculate_log_likelihood(y, X, self.betas)
            self.log_likelihood[it] = log_likelihood
            if self._check_tolerance(it):
                return 'Tolerance achieved', it
        return 'Max iter exceeded', it

    def _calculate_hessian(self, X, betas):
        """ calculate hessian """
        hessian = -X.T @ np.diag(self._get_b_double_prime(X, betas)) @ X
        return hessian

    def _do_gradient_descent(self, X, y):
        """ fit using gradient descent """
        self.betas = np.zeros(X.shape[1])
        for it in tqdm(range(self.max_iteration)):
            gradient = self._get_score_function(X, y, self.betas)
            self.betas = self.betas - self.learning_rate * gradient
            log_likelihood = self._calculate_log_likelihood(y, X, self.betas)
            self.log_likelihood[it] = log_likelihood
            if self._check_tolerance(it):
                return 'Tolerance achieved', it
        return 'Max iter exceeded', it

    def _calculate_log_likelihood(self, y, X, betas):
        """ return log likelihood """
        # avoid log(0)
        epsilon = 0.000001
        mu = self._get_mu(X, self.betas)
        LL = np.sum(y @ np.log(mu+epsilon) + (1 - y) @ np.log(1 - mu+epsilon))
        return LL

    @staticmethod
    def _get_mu(X, betas):
        """ mu = g(predicted_log_odds) """
        mu = 1 / (1 + np.exp(-X@betas))
        return mu

    @staticmethod
    def _get_b_double_prime(X, betas):
        """ get b''(x) """
        b_double_prime = np.exp(X@betas) / (1 + np.exp(X@betas)) ** 2
        return b_double_prime

    def _get_score_function(self, X, Y, betas):
        """ calculate score, derivative of log likelihood """
        mu = self._get_mu(X, self.betas)
        score = -(Y - mu) @ X
        return score


class LinearRegression(LogisticRegression):
    """
    inherits LogisticRegression to fit linear regression model
    only change LL, mu and score function methods
    """

    def _calculate_log_likelihood(self, y, X, betas):
        LL = -(y - X@betas).T @ (y - X@betas)
        return LL

    @staticmethod
    def _get_mu(X, betas):
        mu = X@betas
        return mu

    def _get_score_function(self, X, y, betas):
        mu = self._get_mu(X, betas)
        score = -(y - mu) @ X
        return score

    @staticmethod
    def _get_b_double_prime(X, betas):
        """ get b''(x) """
        b_double_prime = np.ones(X.shape[0])
        return b_double_prime

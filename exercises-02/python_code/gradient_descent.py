import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, fit_intercept=False, max_iteration=8000, tol=1e-8, learning_rate=0.001):
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
        self.log_likelihood = None
        self.fit_intercept = fit_intercept
        self.betas_it = []
        self.max_iteration = max_iteration
        self.tol = tol
        self.learning_rate = learning_rate
        self.log_likelihood = []

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: array
            independent variable
        y: array
            dependant variable

        Returns
        -------
        betas: array
            array of betas g(mu) = x.T@betas
        """
        if self.fit_intercept:
            X = self._make_intercept(X)
        betas = self._do_gradient_descent(X, y)
        return betas

    @staticmethod
    def _make_intercept(X):
        """

        Parameters
        ----------
        X: array
            independent variable

        Returns
        -------
        bigger_X: array
            same array with column of ones

        """
        column_ones = np.ones(X.shape[0])[:, None]
        bigger_X = np.hstack((column_ones, X))
        return bigger_X

    def predict(self, X):
        """

        Parameters
        ----------
        X: array
            independent variable

        Returns
        -------
        predictions: array
            predicted values
        """
        if self.fit_intercept:
            X = self._make_intercept(X)
        prediction = self._get_mu(X @ self.betas)
        prediction = (prediction > 0.5).astype(int)
        return prediction

    def _do_gradient_descent(self, X, Y):
        """

        Parameters
        ----------
        X: array
            independent variable
        Y: array
            dependant variable

        Returns
        -------

        """
        self.betas = np.zeros(X.shape[1])

        for it in tqdm(range(self.max_iteration)):
            #predefine array length instead...
            self.betas_it.append(self.betas)
            predicted_log_odds = X @ self.betas
            gradient = self._get_score_function(X, Y, predicted_log_odds)
            self.betas = self.betas - self.learning_rate * gradient
            log_likelihood = self._calculate_log_likelihood(Y, predicted_log_odds)
            self.log_likelihood.append(log_likelihood)
            log_likelihood_percent_change = np.abs(
                (self.log_likelihood[it] - self.log_likelihood[it - 1]) / self.log_likelihood[
                    it - 1])
            if log_likelihood_percent_change < self.tol and it > 1:
                return 'tolerance achieved'
        return 'max iterations exceeded'

    def _calculate_log_likelihood(self, y, predicted_log_odds):
        """

        Parameters
        ----------
        y: array
            dependent variable
        predicted_log_odds: array
            X.T@betas

        Returns
        -------
        LL: log likelihood
        """
        mu = self._get_mu(predicted_log_odds)
        LL = np.sum(y @ np.log(mu) + (1 - y) @ np.log(1 - mu))
        return LL

    @staticmethod
    def _get_mu(predicted_log_odds):
        """

        Parameters
        ----------
        predicted_log_odds: array
            X.T@betas

        Returns
        -------
        mu: array
            mu = g(predicted_log_odds)
            predicted

        """
        mu = 1 / (1 + np.exp(-predicted_log_odds))
        return mu

    def _get_score_function(self, X, Y, predicted_log_odds):
        """

        Parameters
        ----------
        X: array
            independent variable
        Y: array
            dependent variable
        predicted_log_odds: array
            X.T@betas


        Returns
        -------

        """
        mu = self._get_mu(predicted_log_odds)
        score = -(Y - mu) @ X
        return score


class LinearRegression(LogisticRegression):
    """
    inherits LogisticRegression to fit linear regression model
    only change LL, mu and score function methods
    """
    def _calculate_log_likelihood(self, y, predicted_odds):
        LL = -(y - predicted_odds).T @ (y - predicted_odds)
        return LL

    @staticmethod
    def _get_mu(predicted_odds):
        mu = predicted_odds
        return mu

    def _get_score_function(self, X, Y, predicted_log_odds):
        mu = self._get_mu(predicted_log_odds)
        score = -(Y - mu) @ X
        return score

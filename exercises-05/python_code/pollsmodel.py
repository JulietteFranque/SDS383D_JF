import numpy as np
from scipy.stats import multivariate_normal, gamma, norm, bernoulli, truncnorm
from tqdm import tqdm
import pandas as pd


class Model:
    def __init__(self, X, y, n_iter=1000):
        self.X = X
        self.y = y
        self.P = self.X[0].shape[1]
        self.S = len(self.X)
        self.n_iter = n_iter
        self.traces = {'betas': np.zeros((self.n_iter, self.S, self.P)), 'tau_squared': np.zeros(self.n_iter),
                       'm': np.zeros((self.n_iter, self.P))}
        self.tau_squared = 1
        self.betas = np.ones((self.P, self.S))
        self.m = np.ones(self.P)
        self.z = [np.random.choice([0, 1], size=len(x)) for x in self.y]
        self.taus = []

    def _update_betas(self):
        for state in range(self.S):
            cov = np.linalg.inv(self.tau_squared * np.eye(self.P) + self.X[state].T @ self.X[state])
            mean = cov @ (1/self.tau_squared * self.m + self.X[state].T@self.z[state])
            self.betas[:, state] = multivariate_normal.rvs(mean=mean, cov=cov)

    def _update_tau_squared(self):
        alpha = (self.S+1)/2
        beta = 0.5 * (((self.betas - self.m[:, None])**2).sum() + 1)
        self.tau_squared = 1/gamma.rvs(a=alpha, scale=1/beta)

    def _update_m(self):
        cov = self.tau_squared / self.S * np.eye(self.P)
        mean = cov/self.tau_squared @ self.betas.sum(axis=1)
        self.m = multivariate_normal.rvs(mean=mean, cov=cov)

    def _update_z(self):
        self.z = []
        for state in range(self.S):
            lower_bound = [-np.inf if x == 0 else 0 for x in self.y[state]]
            upper_bound = [np.inf if x == 1 else 0 for x in self.y[state]]
            self.z.append(truncnorm.rvs(lower_bound, upper_bound, loc=0, scale=1))

    def fit(self):
        for iter in tqdm(range(self.n_iter)):
            self._update_tau_squared()
            self._update_betas()
            self._update_m()
            self._update_z()
            self.taus.append(self.tau_squared)

    def _update_traces(self):
        pass

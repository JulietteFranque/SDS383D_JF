from scipy.stats import norm , multivariate_normal
#super().__init__(X, y, lam=lam, K=K, m=m, d=d, eta=eta, fit_intercept=fit_intercept)


def metropolis_hastings(old):
    new = norm(loc=old, scale=1)




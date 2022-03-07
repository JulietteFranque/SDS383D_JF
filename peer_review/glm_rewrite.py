import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # add progress bar


class LogisticRegression:
    def __init__(self):
        self.log_liks = []
        self.beta = None
        # good practice to define attributes in init even if you update them in another method

    @staticmethod  # when method does not use attributes of the class, use @staticmethod decorator
    def standardize(X):
        """standardize/scale X"""
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        return (X - means) / stds

    @staticmethod
    def binomial_link_function(nu):
        """link function from binomial (sigmoid)"""
        return 1 / (1 + np.exp(-nu))

    @staticmethod
    def b_theta_binomial(theta):
        """b_theta from binomial"""
        return np.log(1 + np.exp(theta))

    def log_likelihood(self, X, y, beta):
        """log likelihood with generic parameters"""
        return np.sum((np.array(y) * (X @ beta) - self.b_theta_binomial(X @ beta)))

    def score_function(self, X, y, beta):
        """for the canonical link function, score function simplifies"""
        nu = X @ beta
        mu = self.binomial_link_function(nu)
        return X.T @ (y - mu)

    def hessian_function(self, X, beta):
        """for the canonical link function, hessian function simplifies"""
        nu = X @ beta
        mu = self.binomial_link_function(nu)
        return X.T @ (np.diag(mu * (1 - mu)) @ X)

    def fit(self, X, y, method='newton', max_iter=1000, add_intercept=True, thresh=1e-6, **kwargs):
        X = self.standardize(X)
        if add_intercept:  # also do this in main fit method to avoid repetition
            X.insert(0, 1, np.ones(len(X)))
        if method == 'newton':
            self.fit_glm_newton(X, y, max_iter, thresh, **kwargs)
        elif method == 'gradient descent':
            self.fit_glm(X, y, max_iter, thresh, **kwargs)
        else:
            print('method should be newton or gradient descent')

    def fit_glm(self, X, y, max_iter, thresh, lr=0.001):
        """generalized function to fit a glm given a link function, b(theta) and data"""
        # if add_intercept: # do this in main fit method to avoid repetition
        #  X.insert(0, 1, np.ones(len(X)))
        # maybe preallocate this to NAs for the max_iter value
        self.beta = np.random.randn(size=X.shape[1])
        for it in tqdm(range(max_iter)):
            ll = self.log_likelihood(X, y, self.beta) # don't think there should be a - here?
            self.log_liks.append(ll)
            if self.check_tolerance(thresh, it):  # this will retun True or false
                print('Tolerance ok')
                break
            grad = self.score_function(X, y, self.beta)
            self.beta += lr * grad

    # since you are using this line in both GD and newton, you can make a method so that you don't have to repeat code
    def check_tolerance(self, thresh, it):
        if abs(self.log_liks[it] - self.log_liks[it - 1]) / abs(self.log_liks[
                                                                    it - 1]) <= thresh and it > 0:
            return True
        return False

    def fit_glm_newton(self, X, y, max_iter, thresh, lr=1):
        """generalized function to fit a glm given a link function, b(theta) and data"""
        # iter = 0 better to use for loop so you don't need to increment iter.
        self.log_liks = []  # maybe preallocate this to NAs for the max_iter value
        self.beta = np.zeros(X.shape[1])
        # while True and iter < max_iter: I don't think that "while True" does anything
        # I used a for loop because it works with tqdm progress bar, but while loop is okay too
        for it in tqdm(range(max_iter)):
            ll = self.log_likelihood(X, y, self.beta)
            self.log_liks.append(ll)
            if self.check_tolerance(thresh, it):  # this will return True or false
                print('Tolerance ok')
                break
            hessian = self.hessian_function(X, self.beta)
            grad = self.score_function(X, y, self.beta)
            self.beta += lr * np.linalg.inv(hessian) @ grad  # added lr
            # ll_old = ll you can use your log_likes list instead
        # iter += 1
    # return beta, log_liks it is nice to make beta and log_liks attributes, that way you can access them easily.


def main():
    """main method"""
   # np.random.seed(1)
   # data = pd.read_csv("data/wdbc.csv", header=None).iloc[:, 1:12]
    data = pd.read_csv("../data/wdbc.csv", header=None).iloc[:, 1:12]#for me to debug
    X = data.iloc[:, 1:]
    y = [1 if y == "M" else 0 for y in data.iloc[:, 0]]
    clf = LogisticRegression()
    clf.fit(X, y, method='newton', lr =.1, max_iter=100000)
    print(clf.beta)
    simple_plot(clf.log_liks, "log_lik_plot.png")

    # preds = link_function(X@beta)
    # sum((preds-y)**2)/len(y)


def simple_plot(series, filename):
    """simple plotting utility"""
    plt.plot(np.arange(0, len(series)), series)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    main()

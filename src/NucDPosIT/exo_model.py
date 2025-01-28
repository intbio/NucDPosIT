from sklearn.base import BaseEstimator 
import numpy as np
import scipy as spy


class ExoModel(BaseEstimator):
    def __init__(self, l0=147, reg_koef = 0.18, max_iter=2000):
        self.scope = (-101, 101)
        self.opt_scope = (-50, 51)
        self.max_iter = max_iter
        self.reg_koef = reg_koef
        self.l0 = l0
        
    def __loss(self, e, y, reg):
        res = np.sum((y - np.convolve(e, e)) ** 2) + reg * np.sum(e ** 2)
        return res
        
    def fit(self, X, y=None, *args, **kwargs):
        digesting_errors = X - self.l0
        self.X_ = digesting_errors
        hsit = np.histogram(digesting_errors, bins=np.arange(*self.scope), density=True)[0]
        bounds = [[0, 1] for i in range((len(hsit) + 1) // 2)]
        optimization = spy.optimize.dual_annealing(self.__loss, bounds, args=(hsit, self.reg_koef), maxiter=self.max_iter, *args, **kwargs)
        optimization.x /= optimization.x.sum()
        self.optimization_ = optimization
        return self

    def predict(self, X):
        predictions = np.zeros_like(X, dtype=float)
        nonzero_ind = (X >= -50) & (X <= 50)
        predictions[nonzero_ind] = em.optimization_.x[X[nonzero_ind] + 50]
        return predictions
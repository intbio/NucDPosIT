from nucleosomes import Nucleosome
from sklearn.base import BaseEstimator 
from sklearn.utils.validation import check_is_fitted
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
    
    def load_probs(self, probs):
        self.optimization_ = spy.optimize.OptimizeResult(x=probs, message='loaded data')
        self.is_fitted_ = True
        
        
    def fit(self, X, y=None, *args, **kwargs):
        digesting_errors = X - self.l0
        self.X_ = digesting_errors
        hsit = np.histogram(digesting_errors, bins=np.arange(*self.scope), density=True)[0]
        bounds = [[0, 1] for i in range((len(hsit) + 1) // 2)]
        optimization = spy.optimize.dual_annealing(self.__loss, bounds, args=(hsit, self.reg_koef), maxiter=self.max_iter, *args, **kwargs)
        optimization.x /= optimization.x.sum()
        self.optimization_ = optimization
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        predictions = np.zeros_like(X, dtype=float)
        nonzero_ind = (X >= -50) & (X <= 50)
        predictions[nonzero_ind] = self.optimization_.x[X[nonzero_ind] + 50]
        return predictions
    
    def digest(self, size):
        return np.random.choice(np.arange(*self.opt_scope), size=size, p=self.optimization_.x).astype(int)
    
    def digest_nucs(self, size, dyad=0, id_='nuc'):
        nucleosomes = [0] * size
        starts, ends = self.digest(size), self.digest(size)
        for i in range(len(nucleosomes)):
            cur_start, cur_end = starts[i], ends[i]
            nucleosomes[i] = Nucleosome(dyad, cur_start, cur_end, id_)
        return nucleosomes
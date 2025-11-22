# from nucleosomes import Nucleosome
from sklearn.base import BaseEstimator 
from sklearn.utils.validation import check_is_fitted
import numpy as np
import scipy as spy



class ExoModel(BaseEstimator):
    def __init__(self, l0=147, reg_koef = 0.18, max_iter=2000):
        self.max_iter = max_iter
        self.__reg_koef = reg_koef
        self.l0 = l0
        
    @property
    def reg_koef(self):
        return self.__reg_koef
    
    @reg_koef.setter
    def reg_koef(self, new_koef):
        self.__reg_koef = new_koef        
        
    def __loss(self, e, y, reg):
        res = np.sum((y - np.convolve(e, e)) ** 2) + reg * np.sum(e ** 2)
        return res
    
    def load_probs(self, probs):
        self.optimization_ = spy.optimize.OptimizeResult(x=probs, message='loaded data')
        self.is_fitted_ = True        
        
    def fit(self, hsit, y=None, *args, **kwargs):
        self.X_ = hsit
        bounds = [[0, 1] for i in range((len(hsit) + 1) // 2)]
        x0 = np.random.uniform(size=(len(hsit) + 1) // 2)
        optimization = spy.optimize.dual_annealing(self.__loss, bounds, x0=x0, args=(self.X_, self.reg_koef), maxiter=self.max_iter, *args, **kwargs)
        optimization.x /= optimization.x.sum()
        self.optimization_ = optimization
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        predictions = np.zeros_like(X, dtype=float)
        out_size = (len(self.X_) + 1) // 2
        nonzero_ind = (X >= -out_size // 2) & (X <= out_size // 2)
        predictions[nonzero_ind] = self.optimization_.x[X[nonzero_ind] + out_size // 2]
        return predictions        
    
    def digest(self, size):
        return np.random.choice(np.arange(*self.opt_scope), size=size, p=self.optimization_.x).astype(int)
    
    # def digest_nucs(self, size, dyad=0, id_='nuc'):
    #     nucleosomes = [0] * size
    #     starts, ends = self.digest(size), self.digest(size)
    #     for i in range(len(nucleosomes)):
    #         cur_start, cur_end = starts[i], ends[i]
    #         nucleosomes[i] = Nucleosome(dyad, cur_start, cur_end, id_)
    #     return nucleosomes
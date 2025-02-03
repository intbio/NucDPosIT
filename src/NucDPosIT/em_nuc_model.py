from abc import ABC, abstractmethod
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import scipy as spy
from collections import Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 


class CoordinateProbsMixin:
    def __init__(self, fit_res):
        self.__fit_res = fit_res
        
    @property
    def fit_res(self):
        return self.__fit_res
    
    def __get_left_prob(self, left_cord, dyad_pos):
        i = dyad_pos - left_cord - 23
        if i < 0 or i >= len(self.fit_res):
            return 0
        return self.fit_res[i]

    def get_left_prob(self, left_cord, dyad_pos):
        return np.array(
            list(map(lambda x: self.__get_left_prob(x, dyad_pos), left_cord))
        )

    def __get_right_prob(self, right_cord, dyad_pos):
        i = right_cord - dyad_pos - 23
        if i < 0 or i >= len(self.fit_res):
            return 0
        return self.fit_res[i]


    def get_right_prob(self, right_cord, dyad_pos):
        return np.array(
            list(map(lambda x: self.__get_right_prob(x, dyad_pos), right_cord))
        )
    
    def get_tlen_prob(self, left_cord, right_cord, dyad_pos):
        return self.get_left_prob(left_cord, dyad_pos) * self.get_right_prob(right_cord, dyad_pos)

# -----------------------------------------------------------------------------------


class AbstractEMStrategy(ABC):
    def __init__(self, n_nucs, positions0, weights0, max_iter):
        self.n_nucs = n_nucs
        self.positions = positions0
        self.weights = weights0
        self.max_iter = max_iter
        self.__gij = None
        self.__m = None
        self.__nofreads = None
        
    @property
    def gij(self):
        return self.__gij
    
    @gij.setter
    def gij(self, new_gij):
        self.__gij = new_gij
        
    @property
    def m(self):
        return self.__m
    
    @m.setter
    def m(self, new_m):
        self.__m = new_m
        
    @property
    def nofreads(self):
        return self.__nofreads
    
    @nofreads.setter
    def nofreads(self, new_nofreads):
        self.__nofreads = new_nofreads
    
    
class BaseEMStrategy(CoordinateProbsMixin, AbstractEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, max_iter, fit_res):
        AbstractEMStrategy.__init__(self, n_nucs, positions0, weights0, max_iter)
        CoordinateProbsMixin.__init__(self, fit_res)
        self.offset = None

    def E_step(self, left_cords, right_cords):
        X = np.zeros((left_cords.shape[0], self.n_nucs))
        for j in range(self.n_nucs):
            X[:, j] = self.get_tlen_prob(left_cords, right_cords, self.positions[j]) * self.weights[j]
        p_x = self.weights @ X.T + 1e-50
        self.gij = X / (p_x.reshape(-1, 1))
        self.gij = self.gij / (self.gij.sum(axis=1).reshape(-1, 1) + 1e-50)
        
        
    def set_m_matrix(self, left_cords, right_cords):
        max_right_cord = right_cords.max() + 200
        min_left_cord = left_cords.min() - 200
        nofreads = left_cords.shape[0]
        m = np.zeros((nofreads, max_right_cord - min_left_cord + 1))
        for j, dyad in enumerate(range(min_left_cord, max_right_cord + 1)):
            m[:, j] = self.get_left_prob(left_cords, dyad) * self.get_right_prob(right_cords, dyad)
        m[m == 0] = np.log(m[m == 0] + 1e-50)
        return m
    
    def run_strategy(self, left_cords, right_cords):
        self.gij = np.zeros((left_cords.size, self.n_nucs))
        self.m = self.set_m_matrix(left_cords, right_cords)
        self.offset = left_cords.min() - 200
        self.nofreads = left_cords.shape[0]
        
    def detect_outliers_index(self):
        return np.arange(self.nofreads)[np.all(self.gij == 0, axis=1)]
    
    def detect_lowess_out_index(self, left_cords, right_cords):
        tlen_probs = self.get_tlen_prob()
    
    def cluster_occupancy(self):
        class_occ = Counter(np.argmax(self.gij, axis=1))
        return np.array([class_occ[label] for label in range(self.n_nucs)])
    

class EMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(left_cords, right_cords)
            self.M_step()

    def M_step(self):
        self.positions = np.argmax(self.gij.T @ self.m, axis=1) + self.offset
        self.weights = self.gij.sum(axis=0) / self.nofreads
        self.weights /= self.weights.sum()
        

class EMAlgorythmAdditionComp(EMAlgorythm):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, fit_res, max_iter)
        
    def E_step(self, left_cords, right_cords):
        super().E_step(left_cords, right_cords)
        cluster_occ = self.cluster_occupancy()
        outliers_ind = self.detect_outliers_index()
        # print(outliers_ind.size, cluster_occ.max() * 0.66)
        if outliers_ind.size > self.nofreads * 0.5:
            self.add_component(outliers_ind, left_cords, right_cords)
            
    def add_component(self, out_ind, left_cords, right_cords):
        self.n_nucs += 1
        out_left, out_right = left_cords[out_ind], right_cords[out_ind]
        new_gij_column = np.zeros((self.gij.shape[0], 1))
        new_gij_column[out_ind] = 1
        self.gij = np.hstack((self.gij, new_gij_column))
        print("add component")

            
        


class StochasticEMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        polynom_modeling = np.zeros_like(self.gij)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(left_cords, right_cords)
            self.S_step(polynom_modeling)
            self.M_step(polynom_modeling)

    def S_step(self, polynom_modeling):
        for i in range(self.nofreads):
            polynom_modeling[i, :] = spy.stats.multinomial.rvs(1, self.gij[i, :])
        self.weights = polynom_modeling.sum(0) / self.nofreads
        self.weights /= self.weights.sum()

    def M_step(self, polynom_modeling):
        self.positions = np.argmax(polynom_modeling.T @ self.m, axis=1) + self.offset
        



class EMNucModel(BaseEstimator, ClusterMixin):
    def __init__(self, n_nucs, cluster_strategy, fit_res, max_iter=1000):
        self.n_nucs = n_nucs
        self.max_iter = max_iter
        self.cluster_strategy = cluster_strategy
        self.max_iter = max_iter
        self.fit_res = fit_res

    def __validate_weights(self, weights):
        if weights.size != self.n_nucs:
            raise AttributeError("size of weights should be equal n_nucs")
        return weights

    def init_params(self, X):
        kmeans = KMeans(self.n_nucs)
        kmeans.fit(X)
        self.positions_ = kmeans.cluster_centers_.sum(1).astype(int) // 2
        cluster_counts = Counter(kmeans.predict(X))
        w0 = np.array(
            [
                cluster_counts[cluster_label]
                for cluster_label in range(kmeans.n_clusters)
            ]
        )
        self.weights_ = w0 / w0.sum()

    def fit(self, X, weights0=None, positions0=None, y=None):
        self.X_ = X
        self.y_ = y

        if weights0 is not None and positions0 is not None:
            self.weights_, self.positions_ = weights0, positions0
        elif weights0 is None and positions0 is None:
            self.init_params(X)
        else:
            raise AttributeError(
                "initial positions and weights should be passed both or not pased at all"
            )

        if self.cluster_strategy == "EM":
            self.model = EMAlgorythm(
                self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter
            )
        elif self.cluster_strategy == "SEM":
            self.model = StochasticEMAlgorythm(
                self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter
            )
        elif self.cluster_strategy == 'ADDEM':
            self.model = EMAlgorythmAdditionComp(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter)
        else:
            raise AttributeError("cluster strategy can be SEM or EM")

        try:
            self.model.run_strategy(self.X_[:, 0], self.X_[:, 1])
        except Exception as e:
            raise e
        else:
            self.positions_, self.weights_ = self.model.positions, self.model.weights
            self.labels_ = self.predict(X)
            self.is_fitted_ = True
        return self

    def fit_predict(self, X, weights0, positions0, y=None):
        self.fit(X, weights0, positions0, y)
        return self.labels_

    def predict(self, X):
        prob_matrix = self.predict_matrix(X)
        return np.argmax(prob_matrix, axis=0)

    def predict_matrix(self, X):
        left_cords, right_cords = X[:, 0], X[:, 1]
        gij = np.zeros((left_cords.size, self.n_nucs))
        self.model.E_step(left_cords, right_cords)
        return gij.T

    def score(self):
        prob_matrix = self.predict_matrix(self.X_)
        return np.sum(np.log(self.weights_ @ prob_matrix + 1e-10))
    
    

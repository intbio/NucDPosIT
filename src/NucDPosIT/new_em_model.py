from abc import ABC, abstractmethod
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import scipy as spy
from collections import Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
from sklearn.utils.validation import check_is_fitted
from functools import lru_cache


class AbstractCoordinateProbsMixin(ABC):
    def __init__(self, fit_res):
        self.__fit_res = fit_res
        
    @property
    def fit_res(self):
        return self.__fit_res
    
    @abstractmethod
    def get_tlen_prob(self, left_cord, right_cord, dyad_pos):
        pass

    
class CoordinateProbsMixin(AbstractCoordinateProbsMixin):
    def __init__(self, fit_res):
        super().__init__(fit_res)
    
    @lru_cache(1024)
    def __get_left_prob(self, left_cord, dyad_pos):
        i = dyad_pos - left_cord - 23
        if i < 0 or i >= len(self.fit_res):
            return 0
        return self.fit_res[i]

    def get_left_prob(self, left_cord, dyad_pos):
        return np.array(
            list(map(lambda x: self.__get_left_prob(x, dyad_pos), left_cord))
        )
    
    @lru_cache(1024)
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
        self.__X = None
        
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
        
    @property
    def X(self):
        return self.__X
    
    @X.setter
    def X(self, newX):
        self.__X = newX
    
    
class BaseEMStrategy(CoordinateProbsMixin, AbstractEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, max_iter, fit_res):
        AbstractEMStrategy.__init__(self, n_nucs, positions0, weights0, max_iter)
        CoordinateProbsMixin.__init__(self, fit_res)
        self.offset = None

    def E_step(self, left_cords, right_cords):
        for j in range(self.n_nucs):
            self.X[:, j] = self.get_tlen_prob(left_cords, right_cords, self.positions[j]) * self.weights[j]
        p_x = self.weights @ self.X.T
        p_x[p_x == 0] = 1e-100
        self.gij = self.X / p_x.reshape(-1, 1)
        self.gij = self.gij / (self.gij.sum(axis=1).reshape(-1, 1) + 1e-100)
        
    def M_step(self):
        self.positions = np.argmax(self.gij.T @ self.m, axis=1) + self.offset
        self.weights = self.gij.sum(axis=0) / self.nofreads
        # self.weights /= self.weights.sum()
        
    def set_m_matrix(self, left_cords, right_cords):
        max_right_cord = right_cords.max() 
        min_left_cord = left_cords.min()
        nofreads = left_cords.shape[0]
        m = np.zeros((nofreads, max_right_cord - min_left_cord + 1))
        for j, dyad in enumerate(range(min_left_cord, max_right_cord + 1)):
            probs = self.get_tlen_prob(left_cords, right_cords, dyad)
            m[:, j] = probs
        m[m == 0] = 1e-100
        m = np.log(m)
        return m
    
    def _init_fields(self, left_cords, right_cords):
        self.gij = np.zeros((left_cords.size, self.n_nucs))
        self.m = self.set_m_matrix(left_cords, right_cords)
        self.offset = left_cords.min()
        self.nofreads = left_cords.shape[0]
        self.X = np.zeros((left_cords.shape[0], self.n_nucs))
        
    def run_strategy(self, left_cords, right_cords):
        self._init_fields(left_cords, right_cords)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(left_cords, right_cords)
            self.M_step()
        
    def detect_outliers_index(self):
        pass
    
    def cluster_occupancy(self):
        class_occ = Counter(np.argmax(self.gij, axis=1))
        return np.array([class_occ[label] for label in range(self.n_nucs)])
    
    def log_score(self, batch):
        left_cords, right_cords = batch[:, 0], batch[:, 1]
        X = np.zeros((left_cords.size, self.n_nucs))
        for j in range(self.n_nucs):
            X[:, j] = self.get_tlen_prob(left_cords, right_cords, self.positions[j])
        p_x = X @ self.weights.T
        p_x[p_x == 0] = 1e-100
        score = np.sum(np.log(p_x))
        return score
    
    def predict(self, X):
        tlen_probs = np.zeros((self.n_nucs, X.shape[0]))
        starts, stops = X[:, 0], X[:, 1]
        for i, dyad_pos in enumerate(self.positions):
            tlen_probs[i, :] = self.get_tlen_prob(starts, stops, dyad_pos)
        return (self.weights @ tlen_probs).reshape(-1, 1)
  
    
class AdditionCompStrategy(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, fit_res, max_iter)
        
    def E_step(self, left_cords, right_cords):
        super().E_step(left_cords, right_cords)
        cluster_occ = self.cluster_occupancy()
        outliers_ind = self.detect_outliers_index()
        if outliers_ind.size > self.nofreads * 0.5:
            self.add_component(outliers_ind, left_cords, right_cords)
            
    def add_component(self, out_ind, left_cords, right_cords):
        self.n_nucs = self.n_nucs + 1
        out_left, out_right = left_cords[out_ind], right_cords[out_ind]
        new_gij_column = np.zeros((self.gij.shape[0], 1))
        new_gij_column[out_ind] = 1
        self.gij = np.hstack((self.gij, new_gij_column))
        print("add component")
        
    def detect_outliers_index(self):
        
        return outliers_ind
        

class StochasticEMStrategy(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, fit_res, max_iter)
        self.polynom_modeling = None
        
    def S_step(self):
        for i in range(self.nofreads):
            gij_row = np.absolute(self.gij[i, :])
            multi_probs = np.random.multinomial(1, gij_row)
            self.polynom_modeling[i, :] = multi_probs
        self.weights = self.polynom_modeling.sum(0) / self.nofreads
        self.weights /= self.weights.sum()

    def M_step(self):
        self.positions = np.argmax(self.polynom_modeling.T @ self.m, axis=1) + self.offset
        
    def _init_fields(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        self.polynom_modeling = np.zeros_like(self.gij)

        
class DropingStochasticEMStrategy(StochasticEMStrategy):
    def __init__(self, max_comp, alpha, positions0, weights0, fit_res, max_iter=500):
        super().__init__(max_comp, positions0, weights0, fit_res, max_iter)
        self.alpha = alpha
        self.min_incluster = None
        self.drop_counter = 100
        
    def _init_fields(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        self.min_incluster = int((self.alpha * left_cords.size))
            
    def E_step(self, left_cords, right_cords):
        super().E_step(left_cords, right_cords)
        cluster_occ = self.cluster_occupancy()
        argmin_cluster_occ = np.argmin(cluster_occ)
        self.drop_counter -= 1
        if self.drop_counter < 0 and not np.all(cluster_occ == 0) and cluster_occ[argmin_cluster_occ] < self.min_incluster:
            self.__drop_component(argmin_cluster_occ, left_cords, right_cords)
            
            
    def __drop_component(self, drop_i, left_cords, right_cords):
        self.drop_counter = 20
        self.n_nucs -= 1
        self.gij = np.delete(self.gij, drop_i, 1)
        drop_tlens_index = self.polynom_modeling[:, drop_i] == 1
        best_component = np.argmin(self.cluster_occupancy())
        self.polynom_modeling[drop_tlens_index, best_component] = 1
        self.polynom_modeling = np.delete(self.polynom_modeling, drop_i, 1)
        self.X = np.delete(self.X, drop_i, 1)
        self.positions = np.delete(self.positions, drop_i)
        self.weights = np.delete(self.weights, drop_i)
        super().E_step(left_cords, right_cords)
        
    def cluster_occupancy(self):
        return self.polynom_modeling.sum(0)
        
    


# ----------------------------------------------------------------------------------------------
class EMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
      
        
class AdditionEMAlgorythm(AdditionCompStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)
        
    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        

class StochasticEMAlgorythm(StochasticEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(left_cords, right_cords)
            self.S_step()
            self.M_step()  
            
            
class DropingStochasticEMAlgorythm(DropingStochasticEMStrategy):
    def __init__(self, max_comp, positions0, weights0, fit_res, max_iter=500, alpha=0.05):
        super().__init__(max_comp, alpha, positions0, weights0, fit_res, max_iter)
        
    def run_strategy(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(left_cords, right_cords)
            self.S_step()
            self.M_step() 
        

class AdditionStochasticEMAlgorythm(StochasticEMAlgorythm, AdditionCompStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, fit_res, max_iter)   
        
    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        
    def add_component(self, out_ind, left_cords, right_cords):
        super().add_component(out_ind, left_cords, right_cords)
        self.polynom_modeling = np.zeros_like(self.gij)
        



# -----------------------------------------------------------------------------------------------------------------------     
class EMNucModel(BaseEstimator, ClusterMixin):
    def __init__(self, n_nucs, cluster_strategy, fit_res, max_iter=1000, drop_treshold=0.05):
        self.n_nucs = n_nucs
        self.max_iter = max_iter
        self.cluster_strategy = cluster_strategy
        self.max_iter = max_iter
        self.fit_res = fit_res
        self.drop_treshold = drop_treshold
        self.positions_ = None
        self.weights_ = None
        
    @property
    def model(self):
        return self.__cluster_strategy_
        
    def __set_algorythm(self):
        if self.cluster_strategy == 'EM':
            self.__cluster_strategy_ = EMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter)
        elif self.cluster_strategy == 'SEM':
            self.__cluster_strategy_ = StochasticEMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter)
        elif self.cluster_strategy == 'DROPSEM':
            self.__cluster_strategy_ = DropingStochasticEMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.drop_treshold, self.fit_res, self.max_iter)
        else:
            raise ValueError()
            
    def __set_initial_params(self, X):
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
        
    def __validate_X(self, X):
        lefts, rights = X[:, 0], X[:, 1]
        return lefts, rights
            
    def fit(self, X, y=None):
        self.X_ = X
        self.y_ = y
        self.__set_initial_params(X)
        self.__set_algorythm()
        lefts, rights = self.__validate_X(X)
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
        self.model.E_step(left_cords, right_cords)
        return self.model.gij.T

    def score(self, X):
        return self.model.log_score(X)
    
    def cluster_occupancy(self):
        check_is_fitted(self)
        return self.model.cluster_occupancy()
        
        
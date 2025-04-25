from func_utils import multinomial_rvs, jit_get_left_prob, jit_get_right_prob

from abc import ABC, abstractmethod
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import scipy as spy
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.utils.validation import check_is_fitted
from functools import lru_cache
import pandas as pd


class AbstractCoordinateProbsMixin(ABC):
    def __init__(self, fit_res):
        self.__fit_res = fit_res
        
    @property
    def fit_res(self):
        return self.__fit_res
    
    @abstractmethod
    def get_tlen_prob(self, left_cords, right_cords, dyad_positions, n_nucs):
        pass

    
class CoordinateProbsMixin(AbstractCoordinateProbsMixin):
    def __init__(self, fit_res):
        super().__init__(fit_res)
    
    def get_left_prob(self, left_cords, dyad_positions, n_nucs):
        return self.__get_prob_matrix(left_cords, dyad_positions)

    def __get_prob_matrix(self, left_cords, dyad_positions, offset=23):
        n_nucs = dyad_positions.shape[0]
        left_cord_matrix = np.repeat(left_cords.reshape(-1, 1), n_nucs, axis=1)
        dyad_matrix = np.repeat(dyad_positions.reshape(1, -1), left_cords.shape[0], axis=0)
        index_matrix = -left_cord_matrix + dyad_matrix - offset
        mask = np.abs(index_matrix) > self.fit_res.shape[0] - 1
        index_matrix[mask] = 0
        left_probs = self.fit_res[index_matrix]
        left_probs[mask] = 0
        return left_probs

    def get_right_prob(self, right_cords, dyad_positions, n_nucs):
        return self.__get_prob_matrix(-right_cords, -dyad_positions)
    
    def get_tlen_prob(self, left_cords, right_cords, dyad_positions):
        n_nucs = dyad_positions.shape[0]
        return self.get_left_prob(left_cords, dyad_positions, n_nucs) * self.get_right_prob(right_cords, dyad_positions, n_nucs)
            
# -----------------------------------------------------------------------------------


class AbstractEMStrategy(ABC):
    def __init__(self, n_nucs, positions0, weights0, max_iter, temp, tol):
        self.n_nucs = n_nucs
        self.positions = positions0
        self.weights = weights0
        self.max_iter = max_iter
        self.__gij = None
        self.__m = None
        self.__nofreads = None
        self.__X = None
        self.__temp = temp
        self.__tol = tol
        
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
        
    @property
    def temp(self):
        return self.__temp
    
    @temp.setter
    def temp(self, new_temp):
        self.__temp = new_temp
    
    @property
    def tol(self):
        return self.__tol
    
    @tol.setter
    def tol(self, new_tol):
        self.__tol = new_tol
    
class BaseEMStrategy(CoordinateProbsMixin, AbstractEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, max_iter, fit_res, temp, tol, tau=0):
        AbstractEMStrategy.__init__(self, n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol)
        CoordinateProbsMixin.__init__(self, fit_res)
        self.offset = None
        self.tau = tau

    def E_step(self, left_cords, right_cords):        
        self.X = self.get_tlen_prob(left_cords, right_cords, self.positions) * self.weights
        p_x = self.weights @ self.X.T
        p_x[p_x == 0] = 1e-100
        self.gij = self.X / p_x.reshape(-1, 1)
        self.gij = self.gij / (self.gij.sum(axis=1).reshape(-1, 1) + 1e-100)

        
    def M_step(self):        
        self.positions = np.argmax(self.gij.T @ self.m, axis=1) + self.offset
        self.weights = self.gij.sum(axis=0) / self.nofreads - self.tau
        if np.any(self.weights < 0):
            self.weights[self.weights < 0] = 0
            self.weights = self.weights / self.weights.sum()

        
    def set_m_matrix(self, left_cords, right_cords):
        max_right_cord = right_cords.max() 
        min_left_cord = left_cords.min()
        nofreads = left_cords.shape[0]
        dyads = np.arange(min_left_cord, max_right_cord + 1)
        m = self.get_tlen_prob(left_cords, right_cords, dyads)
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
        self.__crit = 1000
        for i in range(self.max_iter):
            old_weights = self.weights.copy()
            self.E_step(left_cords, right_cords)
            self.M_step()
            delta_weights = self.__evaluate_weights(old_weights)
            # print(i, delta_weights)
            if delta_weights < self.tol:
                break
                
    def __evaluate_weights(self, old_weights):
        crit = self.temp * np.sum(np.abs(self.weights - old_weights)) + (1 - self.temp) * self.__crit
        self.__crit = crit
        return crit
        
        
    def detect_outliers_index(self):
        pass
    
    def cluster_occupancy(self):
        return self.gij.sum(axis=0)
    
    def log_score(self, batch):
        left_cords, right_cords = batch[:, 0], batch[:, 1]
        X = self.get_tlen_prob(left_cords, right_cords, self.positions)
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
    def __init__(self, n_nucs, positions0, weights0, fit_res, temp, tol, max_iter, tau=0):
        super().__init__(n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)
        
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
        # print("add component")
        
        

class StochasticEMStrategy(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)
        self.polynom_modeling = None
        
    def S_step(self):
        self.polynom_modeling = multinomial_rvs(self.nofreads, self.gij)
        self.weights = self.polynom_modeling.sum(0) / self.nofreads
        self.weights /= self.weights.sum() - self.tau
        if np.any(self.weights < 0):
            self.weights[self.weights < 0] = 0
            self.weights = self.weights / self.weights.sum()

    def M_step(self):
        self.positions = np.argmax(self.polynom_modeling.T @ self.m, axis=1) + self.offset
        
    def _init_fields(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        self.polynom_modeling = np.zeros_like(self.gij)
        
    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)

        
class DropingStochasticEMStrategy(StochasticEMStrategy):
    def __init__(self, max_comp, alpha, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs=max_comp, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)
        self.alpha = alpha
        self.min_incluster = None
        self.drop_counter = 40
        
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
        
        
    


# ----------------------------------------------------------------------------------------------
class EMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)

    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
      
        
class AdditionEMAlgorythm(AdditionCompStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)
        
    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        

class StochasticEMAlgorythm(StochasticEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs=n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)
        
    def M_step(self):
        self.S_step()
        super().M_step() 

    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
            
            
class DropingStochasticEMAlgorythm(DropingStochasticEMStrategy):
    def __init__(self, max_comp, positions0, weights0, fit_res, max_iter, alpha, temp, tol, tau=0):
        super().__init__(max_comp, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, alpha=alpha, tau=tau)
        
    def run_strategy(self, left_cords, right_cords):
        super()._init_fields(left_cords, right_cords)
        for i in range(self.max_iter):
            self.E_step(left_cords, right_cords)
            self.S_step()
            self.M_step() 
        

class AdditionStochasticEMAlgorythm(StochasticEMAlgorythm, AdditionCompStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter, temp, tol, tau=0):
        super().__init__(n_nucs, positions0=positions0, weights0=weights0, max_iter=max_iter, temp=temp, tol=tol, fit_res=fit_res, tau=tau)   
        
    def run_strategy(self, left_cords, right_cords):
        super().run_strategy(left_cords, right_cords)
        
    def add_component(self, out_ind, left_cords, right_cords):
        super().add_component(out_ind, left_cords, right_cords)
        self.polynom_modeling = np.zeros_like(self.gij)
        

# -----------------------------------------------------------------------------------------------------------------------     
class EMNucModel(BaseEstimator, ClusterMixin):
    def __init__(self, n_nucs, cluster_strategy, fit_res, max_iter=1000, drop_treshold=0.05, temp=0.3, tol=0.04, tau=0, positions_=None, weights_=None):
        self.n_nucs = n_nucs
        self.max_iter = max_iter
        self.cluster_strategy = cluster_strategy
        self.max_iter = max_iter
        self.__fit_res = fit_res
        self.drop_treshold = drop_treshold
        self.positions_ = positions_
        self.weights_ = weights_
        self.temp = temp
        self.tol = tol
        self.tau = tau
        
    @property
    def fit_res(self):
        return self.__fit_res
        
    @property
    def model(self):
        return self.__cluster_strategy_
    
    @property
    def scoring_function(self):
        return self.__scoring_function
        
    def __set_algorythm(self):
        if self.cluster_strategy == 'EM':
            self.__cluster_strategy_ = EMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter, self.temp, self.tol, tau=self.tau)
        elif self.cluster_strategy == 'SEM':
            self.__cluster_strategy_ = StochasticEMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter, self.temp, self.tol, tau=self.tau)
        elif self.cluster_strategy == 'DROPSEM':
            self.__cluster_strategy_ = DropingStochasticEMAlgorythm(self.n_nucs, self.positions_, self.weights_, self.fit_res, self.max_iter, self.drop_treshold, self.temp, tol=self.tol, tau=self.tau)
        else:
            raise ValueError()
            
    def __set_initial_params(self, X):
        if self.positions_ is None and self.weights_ is None:
            min_x, max_x = X[:, 0].min(), X[:, 1].max()
            # mids = X.mean(axis=1).reshape(-1, 1)
            # # positions_ = np.linspace(min_x, max_x, self.n_nucs).astype(int)
            positions_ = np.random.uniform(min_x, max_x + 1, self.n_nucs).astype(int)
            # dist_matrx = spy.spatial.distance.cdist(positions_.reshape(-1, 1), mids)
            # cluster_counts = Counter(np.argmin(dist_matrx, axis=0))
            # w0 = np.array(
            #     [
            #         cluster_counts[cluster_label]
            #         for cluster_label in range(self.n_nucs)
            #     ]
            # )
            # weights_ = w0 / w0.sum()
            self.positions_ = positions_
            self.weights_ = np.array([1 / self.n_nucs] * self.n_nucs)

            # kmeans = KMeans(self.n_nucs)
            # kmeans.fit(X)
            # self.positions_ = kmeans.cluster_centers_.sum(1).astype(int) // 2
            # cluster_counts = Counter(kmeans.predict(X))
            # w0 = np.array(
            #     [
            #         cluster_counts[cluster_label]
            #         for cluster_label in range(kmeans.n_clusters)
            #     ]
            # )
            # self.weights_ = w0 / w0.sum() 
        
        
        
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
    
    
class EMNucViewer:
    def __init__(self, start, stop, emnuc_model=None):
        self.start = start
        self.stop = stop
        self.__window_df = self.__make_window_df(emnuc_model)
        self.__stat_df = self.__make_stat_df(emnuc_model)
    
    @property
    def window_df(self):
        return self.__window_df
    
    @window_df.setter
    def window_df(self, new_df):
        self.__window_df = new_df
    
    @property
    def stat_df(self):
        return self.__stat_df
    
    @stat_df.setter
    def stat_df(self, new_df):
        self.__stat_df = new_df
    
    def merge(self, new_model):
        new_viewer = EMNucViewer(self.start, self.stop, new_model)
        new_window_df = new_viewer.window_df
        self.__window_df = pd.concat([new_window_df, self.window_df]) if self.__window_df is not None else new_window_df
        new_stat_df = new_viewer.stat_df
        self.__stat_df = pd.concat([new_stat_df, self.stat_df]) if self.__stat_df is not None else new_stat_df
        return self
    
    def drop_duplicates(self):
        # window_df = self.window_df.drop_duplicates()
        new_stat_df = self.stat_df.drop_duplicates('dyad')
        new_viewer = EMNucViewer(self.start, self.stop)
        new_viewer.window_df, new_viewer.stat_df = self.window_df, new_stat_df
        return new_viewer
        
    def __make_window_df(self, model=None):
        if model is None:
            return None
        batch = model.X_
        prob_matrix = model.model.gij
        window_data = pd.DataFrame(batch, columns=['start', 'stop'])
        window_data['mid'] = (window_data.start + window_data.stop) / 2
        nuc_indxex = np.argmax(model.model.gij, axis=1)
        window_data["dyad"] = model.positions_[nuc_indxex]
        window_data["dyadLH"] = np.max(prob_matrix, axis=1)
        window_data["tempLH"] = model.model.X.sum(axis=1)
        window_data['dyad_prob'] = model.weights_[nuc_indxex]
        return window_data

    def __make_stat_df(self, model):
        if model is None:
            return None
        window_data = self.window_df
        group_size = window_data.groupby("dyad").size().reset_index()
        group_size.name = 'size'
        group_size.rename(columns={0: 'size'}, inplace=True)
        window_data['stat'] = window_data.dyad - window_data.mid
        window_data['stat'] = window_data.groupby('dyad', group_keys=False).stat.apply(lambda x: x  / x.std() )
        window_data['stat'] = window_data['stat'] ** 2
        norm1 = window_data.groupby('dyad').stat.sum()
        norm1.name = 'stat'
        statistics = pd.merge(norm1, group_size, on='dyad').reset_index(drop=True)
        p_values = [1 - spy.stats.chi2.cdf(row.stat, row['size']) for i, row in statistics.iterrows()]
        statistics['p_vals'] = p_values
        
        tmp = pd.DataFrame([model.positions_, model.cluster_occupancy(), model.model.weights], index=['dyad', 'height', 'weight']).T
        tmp = tmp.groupby('dyad', as_index=False).sum()
        statistics = pd.merge(statistics, tmp, on='dyad')
    
        return statistics
    
    def plot(self, nuc_template, std_template, ax=None, legend=True):
        if ax is None:
            fig, ax = plt.subplots()
        x, coverage = self.make_coverage()
        model_coverage = self.model_occ(nuc_template)[1]
        model_std = self.model_std(std_template)[1]
        ax.errorbar(x, model_coverage, yerr=model_std, label='model')
        ax.plot(x, coverage, label='experiment')
        ax.bar(self.stat_df.dyad.to_numpy(), self.stat_df.height.to_numpy(), 10, label='dyads')
        if legend:
            ax.legend()
        return ax

    def make_coverage(self):
        batch = self.window_df[['start', 'stop']].to_numpy()
        x = np.zeros(self.stop - self.start + 400)
        broad_start = self.start - 200
        lefts, rights = batch[:, 0].astype(int), batch[:, 1].astype(int)
        for left, right in zip(lefts, rights):
            x[left - broad_start : right - broad_start + 1] += 1
        return np.arange(self.start, self.stop), x[200 : -200]
    
    def model_occ(self, nuc_template):
        dyads = self.stat_df.dyad.to_numpy()
        dyad_heights = self.stat_df.height.to_numpy()
        return self.__make_model_occ(dyads, nuc_template, dyad_heights)
    
    def model_std(self, nuc_template):
        dyads = self.stat_df.dyad.to_numpy()
        dyad_weights = np.sqrt(self.stat_df.height.to_numpy())
        return self.__make_model_occ(dyads, nuc_template, dyad_weights)
          
    def __make_model_occ(self, dyads, nuc_template, dyad_weights=None):
        start, end = self.start, self.stop
        dyad_weights = np.ones_like(dyads) if dyad_weights is None else dyad_weights
        broad_start, broad_end = start - nuc_template.size, end + nuc_template.size
        x = np.arange(broad_start, broad_end)
        y = np.zeros_like(x, dtype=float)
        for i, dyad in enumerate(dyads):
            start_pos = dyad - nuc_template.size // 2 - broad_start
            end_pos = dyad - broad_start + nuc_template.size // 2
            y[start_pos : end_pos] += nuc_template * dyad_weights[i]
        return x[nuc_template.size : -nuc_template.size], y[nuc_template.size : -nuc_template.size]

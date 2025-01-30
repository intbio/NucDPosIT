from abc import ABC, abstractmethod
import sklearn as sk
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import scipy as spy
from collections import Counter
from tqdm.auto import tqdm


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

# -----------------------------------------------------------------------------------


class AbstractEMStrategy(ABC):
    def __init__(self, n_nucs, positions0, weights0, max_iter):
        self.n_nucs = n_nucs
        self.positions = positions0
        self.weights = weights0
        self.max_iter = max_iter
    
    
class BaseEMStrategy(CoordinateProbsMixin):
    def __init__(self, n_nucs, positions0, weights0, max_iter, fit_res):
        self.n_nucs = n_nucs
        self.positions = positions0
        self.weights = weights0
        self.max_iter = max_iter
        super().__init__(fit_res)

    def E_step(self, gij, b, left_cords, right_cords):
        for j in range(self.n_nucs):
            cur_prob = self.get_left_prob(
                left_cords, self.positions[j]
            ) * self.get_right_prob(right_cords, self.positions[j])
            gij[:, j] = self.weights[j] * cur_prob
            b[j, :] = cur_prob
        gij[:] = gij / (self.weights @ b + 1e-10).reshape(-1, 1)

    def set_m_matrix(self, left_cords, right_cords):
        max_right_cord = right_cords.max()
        min_left_cord = left_cords.min()
        nofreads = left_cords.shape[0]
        m = np.zeros((nofreads, 200 + max_right_cord))
        for j in range(min_left_cord, 200 + max_right_cord):
            m[:, j] = self.get_left_prob(left_cords, j) * self.get_right_prob(
                right_cords, j
            )
        m = np.log(m + 1e-10)
        return m


class EMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        gij = np.zeros((left_cords.size, self.n_nucs))
        b = np.zeros((self.n_nucs, left_cords.size))
        m = self.set_m_matrix(left_cords, right_cords)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(gij, b, left_cords, right_cords)
            self.M_step(gij, m)

    def M_step(self, gij, m):
        self.positions = np.argmax(gij.T @ m, axis=1)
        self.weights = gij.sum(axis=0) / gij.shape[0]


class StochasticEMAlgorythm(BaseEMStrategy):
    def __init__(self, n_nucs, positions0, weights0, fit_res, max_iter=500):
        super().__init__(n_nucs, positions0, weights0, max_iter, fit_res)

    def run_strategy(self, left_cords, right_cords):
        gij = np.zeros((left_cords.size, self.n_nucs))
        b = np.zeros((self.n_nucs, left_cords.size))
        m = self.set_m_matrix(left_cords, right_cords)
        polynom_modeling = np.zeros_like(gij)
        for i in tqdm(range(self.max_iter), total=self.max_iter):
            self.E_step(gij, b, left_cords, right_cords)
            self.S_step(gij, polynom_modeling)
            self.M_step(polynom_modeling, m)

    def S_step(self, gij, polynom_modeling):
        nofreads = polynom_modeling.shape[0]
        for i in range(nofreads):
            polynom_modeling[i, :] = spy.stats.multinomial.rvs(1, gij[i, :])
        self.weights = polynom_modeling.sum(0) / nofreads

    def M_step(self, polynom_modeling, m):
        self.positions = np.argmax(polynom_modeling.T @ m, axis=1)


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
        kmeans = sk.cluster.KMeans(self.n_nucs)
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
        b = np.zeros((self.n_nucs, left_cords.size))
        self.model.E_step(gij, b, left_cords, right_cords)
        return gij.T

    def score(self):
        prob_matrix = self.predict_matrix(self.X_)
        return np.sum(np.log(self.weights_ @ prob_matrix + 1e-10))
    
    

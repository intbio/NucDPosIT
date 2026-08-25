import copy

import numpy as np
import pandas as pd
import torch
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)


class EMModel:
    def __init__(
        self,
        errors,
        dyad_dist,
        min_iter=10,
        max_iter=50,
        nfits = 50,
        reg_coef=0,
        temperature_coef=1,
        tol=1e-3,
        device="cpu",
        alpha=0.5,
    ):
        self.device = device
        self.errors = self._read_errors(errors)
        self.dyad_dist = dyad_dist
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.nfits = nfits
        self.reg_coef = reg_coef
        self.tol = tol
        self.alpha = alpha
        self.temperature_coef = temperature_coef


        self.full_probs = None          # будет хранить матрицу P(dyad | read) для всех возможных dyad
        self.log_full_probs = None      # логарифм от full_probs (используется в m_step)
        self.start_min = None           # для быстрого вычисления индексов
        self.position_count = None  

    @property
    def nreads(self):
        return len(self.starts)

    @property
    def ndyads(self):
        return len(self.dyads)

    def _read_errors(self, path):
        if isinstance(path, str):
            errors = np.loadtxt(path, delimiter=",")
            errors = torch.from_numpy(errors).to(self.device)
        elif isinstance(path, np.ndarray):
            errors = torch.from_numpy(errors).to(self.device)
        elif isinstance(path, torch.Tensor):
            errors = path
        else:
            raise TypeError(f"cannot read path {path}")
        errors = errors / errors.sum()
        return errors

    def __validate_cords(self, cords):
        return cords.to(self.device)

    def __validate_dyads(self, dyads=None):
        if dyads is None:
            dyads = (
                torch.arange(
                    self.starts.min(),
                    self.stops.max() + 1,
                    self.dyad_dist,
                    device=self.device,
                )
                .int()
                .reshape(-1, 1)
            )
        return dyads

    def __validate_weights(self, weights=None):
        if weights is None:
            weights = torch.rand(self.ndyads, dtype=float, device=self.device).reshape(
                -1, 1
            )
            weights /= weights.sum()
        return weights

    def create_probs_matrix(self, dyads):
        if self.full_probs is None:
            self.__create_grid_probs()
        indices = (dyads.flatten() - self.start_min).long()
        return self.full_probs[:, indices]   # (n_reads, n_dyads)

    def __create_grid_probs(self):
        potential_dyads = torch.arange(
            self.starts.min(), self.stops.max() + 1, device=self.device
        ).reshape(-1, 1)

        probs = torch.log(self.create_probs_matrix(potential_dyads) + 1e-50)
        return probs

    def e_step(self):
        with torch.amp.autocast(device_type=self.device):
            self.probs_matrix = self.create_probs_matrix(self.dyads)
            log_probs = torch.log(self.probs_matrix)      # (n_reads, n_dyads)
            log_weights = torch.log(self.weights.T)       # (1, n_dyads) – broadcasting
            logits = (log_probs + log_weights) / self.temperature
            self.Hij = torch.softmax(logits, dim=1)

    def m_step(self):
        with torch.amp.autocast(device_type=self.device):
            self.dyads = (
                self.starts.min()
                + torch.argmax(self.grid_probs.T @ self.Hij, dim=0, keepdim=True).T
            )
            self.weights = self.Hij.sum(0, keepdim=True).T
            self.weights = self.weights / self.weights.sum() - self.reg_coef / self.nreads
            if (self.weights <= 0).any():
                self.delete_components()
            else:
                self.weights /= self.weights.sum()
            if (self.weights == 0).all() or self.weights.isnan().all():
                raise ValueError("all weights = 0. It seems reg_coef is too high")

    def delete_components(self):
        keep_alive_mask = (self.weights > 0).bool().reshape(-1, 1)
        self.weights = self.weights[keep_alive_mask].reshape(-1, 1)
        self.weights /= self.weights.sum()
        self.dyads = self.dyads[keep_alive_mask].reshape(-1, 1)
        self.probs_matrix = self.create_probs_matrix(self.dyads)

    def merge_duplicate_dyads(self):        
        unique_dyads, inverse_indices = torch.unique(
            self.dyads.flatten(), 
            return_inverse=True
        )
        if len(unique_dyads) == len(self.dyads):
            return
            
        new_weights = torch.zeros(
            len(unique_dyads), 
            1, 
            device=self.device, 
            dtype=self.weights.dtype
        )
        
        new_weights.scatter_add_(
            0,                         
            inverse_indices.reshape(-1, 1), 
            self.weights                
        )
        
        self.dyads = unique_dyads.reshape(-1, 1)
        self.weights = new_weights
        self.weights = self.weights / self.weights.sum()
        self.probs_matrix = self.create_probs_matrix(self.dyads)


    def add_component(self):
        items_logLH = self.items_logLH()       
        argmin = items_logLH.argmin()            
        min_start = self.starts[argmin]           
        min_end   = self.stops[argmin]
    
        new_dyad_val = (min_start + min_end) // 2   
        if new_dyad_val in self.dyads:
            return
        new_dyad = new_dyad_val.view(1, 1)         
    
        flat_dyads = self.dyads.flatten()   
        distances = torch.abs(flat_dyads - new_dyad) 
        insert_ind = distances.argmin()
    
        self.dyads = torch.cat([
            self.dyads[:insert_ind],
            new_dyad,
            self.dyads[insert_ind:]
        ], dim=0)
    
        read_counts = self.weights * self.nreads 
        read_counts[insert_ind] -= 1                     

        new_count = torch.tensor([[1]], dtype=read_counts.dtype, device=self.device)   # (1,1)
    
        read_counts_new = torch.cat([
            read_counts[:insert_ind],
            new_count,
            read_counts[insert_ind:]
        ], dim=0) 
    
        self.weights = read_counts_new / self.nreads
        
    def to(self, device):
        self.device = device
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                self.__dict__[name] = val.to(device)
        return self

    def get_params(self):
        params = {
            "L": self.starts,
            "R": self.stops,
            "errors": self.errors,
            "dyad_dist": self.dyad_dist,
            "max_iter": self.max_iter,
            "reg_coef": self.reg_coef,
            "tol": self.tol,
            "dyads": self.dyads,
            "weights": self.weights,
            "device": self.device,
        }
        return params

    def logLH(self):
        loglh = self.items_logLH().sum()
        if self.reg_coef > 0:
            mask = (self.weights > 0).squeeze()
            loglh -= self.reg_coef / self.nreads * torch.log(self.weights[mask]).sum()
        return loglh

    def items_logLH(self):
        mask = (self.weights > 0).squeeze()
        items_loglh = torch.log((self.probs_matrix[:, mask] * self.weights[mask].T).sum(axis=1))
        return items_loglh

    def update_sliding_mean(self, cur_logLH=None):
        cur_logLH = cur_logLH if cur_logLH else self.logLH()
        self.logLH_history.append(cur_logLH)
        if len(self.sliding_mean_history) != 0:
            new_slmean = self.alpha * cur_logLH + self.sliding_mean_history[-1] * (1 - self.alpha)
        else:
            new_slmean = cur_logLH
        self.sliding_mean_history.append(new_slmean)

    def reset_sliding_mean(self):
        self.sliding_mean_history = []
        self.logLH_history = []

    def __create_grid_probs(self):
        """Вычислить один раз full_probs для всех возможных dyad и сохранить в кэш."""
        if self.full_probs is not None:
            return 

        start_min = self.starts.min()
        stop_max = self.stops.max()
        potential_dyads = torch.arange(start_min, stop_max + 1, device=self.device).reshape(-1, 1)
        left_diff = potential_dyads - self.starts.view(1, -1)   # (n_positions, n_reads)
        right_diff = self.stops.view(1, -1) - potential_dyads   # (n_positions, n_reads)
        
        def insert_probs(diff_matrix, errors):
            idx = diff_matrix.round().long()
            idx_clamped = idx.clamp(0, len(errors) - 1)
            result = errors[idx_clamped]  # (n_positions, n_reads)
            invalid = (idx < 0) | (idx >= len(errors))
            result[invalid] = 1e-50
            return result

        L_probs = insert_probs(left_diff, self.errors)
        R_probs = insert_probs(right_diff, self.errors)
        full_probs = (L_probs * R_probs).T  

        self.full_probs = full_probs
        self.log_full_probs = torch.log(full_probs + 1e-50)
        self.start_min = start_min
        self.position_count = potential_dyads.shape[0]



    def fit(self, starts, stops, dyads=None, weights=None, verbose=False):
        with torch.no_grad():
            self.reset_sliding_mean()
            self.starts = self.__validate_cords(starts)
            self.stops = self.__validate_cords(stops)

            self.full_probs = None
            self.log_full_probs = None
            self.start_min = None
            self.position_count = None

            self.dyads = self.__validate_dyads(dyads)
            self.probs_matrix = self.create_probs_matrix(self.dyads)

            self.weights = self.__validate_weights(weights)
            self.log_weights = torch.log(self.weights.T + 1e-50)
            self.dyads_history = []

            best_loglh = -np.inf

            for fit_iter in range(self.nfits):
                self.add_component()
                for i in range(self.max_iter):
                    self.temperature = self.temperature_coef / np.log(i + 2)
                    self.inv_temperature = 1.0 / self.temperature
                    self.e_step()
                    self.m_step()
                    self.delete_components()
                    self.merge_duplicate_dyads()

                    self.dyads_history.append(self.ndyads)

                    cur_lh = self.logLH()
                    self.update_sliding_mean(cur_lh)
                    if cur_lh > best_loglh:
                        best_loglh = cur_lh

                    if len(self.sliding_mean_history) >= 2 and i >= self.min_iter:
                        delta = abs(self.sliding_mean_history[-1] - self.sliding_mean_history[-2])
                        if delta < self.tol:
                            break

            self.e_step()
            return self
        
    def to_df(self):
        df = pd.DataFrame(
            {"start": self.starts.flatten().cpu(), "stop": self.stops.flatten().cpu()}
        )
        df["dyads"] = self.dyads[self.Hij.argmax(1)].cpu()
        df["template|dyad"] = self.Hij.max(1)[0].cpu()
        return df
        


class StochasticEMModel(EMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def m_step(self):
        with torch.amp.autocast(device_type=self.device):
            # Генерируем one-hot сэмплы (float32)
            self.stochastic_res = self.sample_multinomial_vectorized_torch()  # (n_reads, n_dyads)
            # Умножаем транспонированные сэмплы на логарифм вероятностей
            weighted_sum = torch.matmul(self.stochastic_res.T, self.log_full_probs)  # (n_dyads, n_positions)
            max_indices = torch.argmax(weighted_sum, dim=1, keepdim=True)
            self.dyads = self.start_min + max_indices.float()

            # Обновляем веса как сумму сэмплов по reads
            self.weights = self.stochastic_res.sum(0, keepdim=True).T
            self.weights = self.weights / self.weights.sum() - self.reg_coef / self.nreads

            if (self.weights <= 0).any():
                self.delete_components()
            self.weights /= self.weights.sum()
            self.log_weights = torch.log(self.weights.T + 1e-50)

    def sample_multinomial_vectorized_torch(self):
        if self.Hij.isnan().any():
            self.delete_components()
            self.e_step()
            return self.sample_multinomial_vectorized_torch()
        samples = torch.multinomial(self.Hij, num_samples=1)
        return torch.nn.functional.one_hot(
            samples.squeeze(-1), 
            num_classes=self.Hij.shape[-1]
        ).double()
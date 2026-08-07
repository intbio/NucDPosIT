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
        max_iter=2000,
        nfits = 50,
        reg_coef=0,
        temperature_coef=1,
        tol=0.0001,
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

    @lru_cache
    def create_probs_matrix(self, dyads):
        n_dyads = dyads.shape[0]
        n_reads = self.nreads
        left_diff = dyads - self.starts.view(1, -1)
        right_diff = self.stops.view(1, -1) - dyads

        L_index = self.__insert_probs_to_matrix(left_diff, self.errors)
        R_index = self.__insert_probs_to_matrix(right_diff, self.errors)
        return (L_index * R_index).T

    def __create_grid_probs(self):
        potential_dyads = torch.arange(
            self.starts.min(), self.stops.max() + 1, device=self.device
        ).reshape(-1, 1)

        probs = torch.log(self.create_probs_matrix(potential_dyads) + 1e-50)
        return probs

    def __insert_probs_to_matrix(self, idx_matrix, errors):
        idx = idx_matrix.round().long()
        valid_mask = (idx >= 0) & (idx < len(errors))
        result = torch.zeros_like(idx_matrix, dtype=float) + 1e-50
        valid_indices = idx[valid_mask]
        result[valid_mask] = errors[valid_indices]

        return result

    def e_step(self):
        self.probs_matrix = self.create_probs_matrix(self.dyads)
        self.Hij = (self.probs_matrix * self.weights.T) ** (1 / self.temperature) / (self.probs_matrix @ self.weights + 1e-50) ** (1 / self.temperature)
        if self.Hij.isnan().any():
            raise ValueError("Hij has None")

    def m_step(self):
        self.dyads = (
            self.starts.min()
            + torch.argmax(self.grid_probs.T @ self.Hij, dim=0, keepdim=True).T
        )
        self.weights = self.Hij.sum(0, keepdim=True).T
        self.weights = self.weights / self.weights.sum() - self.reg_coef / self.nreads
        self.weights[self.weights < 0] = 0
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
        insert_ind = distances.argmin().item()
    
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
    
    def fit(self, starts, stops, dyads=None, weights=None):
        self.starts = self.__validate_cords(starts)
        self.stops = self.__validate_cords(stops)
        self.dyads = self.__validate_dyads(dyads)
        self.probs_matrix = self.create_probs_matrix(self.dyads)
        self.grid_probs = self.__create_grid_probs()
        self.weights = self.__validate_weights(weights)        
    
        best_lh = -torch.inf
        lh_history = []  
        patience_counter = 0
        ncomponents = []
        
        for i in range(self.nfits):
            if i != 0:
                self.add_component()
            try:
                fit_history = self.fit_sliding_mean(self.min_iter, self.max_iter, self.alpha)  
            except ValueError:
                print(f"{self.ndyads} skip")
                continue
            else:
                lh_history.extend(fit_history['lhloss'])
                
                current_lh = fit_history['model_lh']
                if self.logLH() < fit_history['model_lh']:
                    self.__dict__ = fit_history['model'].__dict__
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                ncomponents.append(self.ndyads)  
                if patience_counter >= 3:
                    break
    
        return {'lh': lh_history, 'nweights': ncomponents}

    def fit_sliding_mean(self, min_iter, max_iter, alpha, history=None):
        assert min_iter < max_iter
        assert alpha > 0
        
        if history is None:
            history = {
                'lhloss': [],
                'sliding_mean': [],
                'success': None,
                'model_lh': -torch.inf
            }
        
        sliding_mean = None
        prev_sliding_mean = -torch.inf
        best_model = None
        best_lh = -torch.inf

        # self.delete_components()
        for i in range(max_iter):
            # Обновляем температуру
            self.temperature = self.temperature_coef / np.log(i + 2) 
            
            self.e_step()
            self.m_step()
            if i % 20 == 0:
                self.merge_duplicate_dyads()
                self.e_step()
                self.m_step()
            
            cur_lh = self.logLH().item()
            history['lhloss'].append(cur_lh)
            
            # Обновляем скользящее среднее
            if sliding_mean is None:
                sliding_mean = cur_lh
            else:
                sliding_mean = alpha * cur_lh + (1 - alpha) * sliding_mean
            history['sliding_mean'].append(sliding_mean)
            
            if cur_lh > best_lh:
                best_lh = cur_lh
                best_model = copy.deepcopy(self)
            
            delta = abs(sliding_mean - prev_sliding_mean)
            prev_sliding_mean = sliding_mean
            
            if delta < self.tol and i > min_iter:
                history['success'] = True
                history['model'] = best_model
                history['model_lh'] = best_lh
                return history
        
        # Если достигли max_iter без остановки
        history['success'] = False
        history['model'] = best_model
        history['model_lh'] = best_lh
        return history
        
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
        self.stochastic_res = self.sample_multinomial_vectorized_torch()
        self.weighted_sum = torch.matmul(self.grid_probs.T, self.stochastic_res)
        max_indices = torch.argmax(self.weighted_sum, dim=0, keepdim=True).T
        self.dyads = self.starts.min() + max_indices
        self.weights = self.stochastic_res.sum(0, keepdim=True).T
        self.weights = self.weights / self.weights.sum() - self.reg_coef / self.nreads
        self.weights[self.weights < 0] = 0
        self.weights /= self.weights.sum()

    def sample_multinomial_vectorized_torch(self):
        if torch.isnan(self.Hij).any():
            raise ValueError("Hij matrix contains nan")
        samples = torch.multinomial(self.Hij, num_samples=1)
        return torch.nn.functional.one_hot(
            samples.squeeze(-1), 
            num_classes=self.Hij.shape[-1]
        ).double()
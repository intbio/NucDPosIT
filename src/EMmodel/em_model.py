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
        with torch.amp.autocast(device_type=self.device):
            self.probs_matrix = self.create_probs_matrix(self.dyads)
            log_probs = torch.log(self.probs_matrix + 1e-50)      # (n_reads, n_dyads)
            log_weights = torch.log(self.weights.T + 1e-50)       # (1, n_dyads) – broadcasting
            logits = (log_probs + log_weights) / self.temperature
            logits_max = logits.max(dim=1, keepdim=True)[0]
            logits_stable = logits - logits_max
            exp_logits = torch.exp(logits_stable)
            self.Hij = exp_logits / (exp_logits.sum(dim=1, keepdim=True) + 1e-50)
            if self.Hij.isnan().any():
                self.delete_components()

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

    def update_sliding_mean(self):
        cur_logLH = self.logLH()
        self.logLH_history.append(cur_logLH)
        if len(self.sliding_mean_history) != 0:
            new_slmean = self.alpha * cur_logLH + self.sliding_mean_history[-1] * (1 - self.alpha)
        else:
            new_slmean = cur_logLH
        self.sliding_mean_history.append(new_slmean)

    def reset_sliding_mean(self):
        self.sliding_mean_history = []
        self.logLH_history = []

    def fit(self, starts, stops, dyads=None, weights=None, verbose=False):
        self.reset_sliding_mean()
        self.starts = self.__validate_cords(starts)
        self.stops = self.__validate_cords(stops)
        self.dyads = self.__validate_dyads(dyads)
        self.probs_matrix = self.create_probs_matrix(self.dyads)
        self.grid_probs = self.__create_grid_probs()
        self.weights = self.__validate_weights(weights)
        self.dyads_history = []
        self.bic_history = []
        convergence_flag = False

        best_loglh = -np.inf
        best_state = None

        for i in range(self.nfits):
            self.add_component()
            for i in range(self.max_iter):
                self.temperature = self.temperature_coef / np.log(i + 2)
                self.e_step()
                self.m_step()
                self.delete_components()
                self.merge_duplicate_dyads()

                self.update_sliding_mean()
                self.dyads_history.append(self.ndyads)

                cur_lh = self.logLH()
                if cur_lh > best_loglh:
                    best_loglh = cur_lh

                if len(self.sliding_mean_history) >= 2 and i >= self.min_iter:
                    delta = abs(self.sliding_mean_history[-1] - self.sliding_mean_history[-2])
                    if delta < self.tol:
                        logger.info(f"Convergence reached at iteration {i} (delta={delta:.6f})")
                        break

        # self.__dict__.update(best_state)
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
        self.stochastic_res = self.sample_multinomial_vectorized_torch()
        self.weighted_sum = torch.matmul(self.grid_probs.T, self.stochastic_res)
        max_indices = torch.argmax(self.weighted_sum, dim=0, keepdim=True).T
        self.dyads = self.starts.min() + max_indices
        self.weights = self.stochastic_res.sum(0, keepdim=True).T
        self.weights = self.weights / self.weights.sum() - self.reg_coef / self.nreads
        if (self.weights <= 0).any():
            self.delete_components()
        self.weights /= self.weights.sum()

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
import copy
import torch
import pandas as pd
import numpy as np


class EMModel:
    def __init__(
        self,
        errors,
        dyad_dist,
        max_iter=500,
        reg_coef=0,
        tol=0.0001,
        device="cpu",
        alpha=0.5,
    ):
        self.device = device
        self.errors = self._read_errors(errors)
        self.dyad_dist = dyad_dist
        self.max_iter = max_iter
        self.reg_coef = reg_coef
        self.tol = tol
        self.alpha = alpha

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
            raise TypeError(f'cannot read path {path}')
        errors  = errors / errors.sum()
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

    def __create_probs_matrix(self, dyads):
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
        ).reshape(
            -1, 1
        ) 

        probs = torch.log(self.__create_probs_matrix(potential_dyads) + 1e-50)
        return probs

    def __insert_probs_to_matrix(self, idx_matrix, errors):
        idx = idx_matrix.round().long()
        valid_mask = (idx >= 0) & (idx < len(errors))
        result = torch.zeros_like(idx_matrix, dtype=float) + 1e-50
        valid_indices = idx[valid_mask]
        result[valid_mask] = errors[valid_indices]

        return result

    def e_step(self):
        self.probs_matrix = self.__create_probs_matrix(self.dyads)
        self.Hij = (self.probs_matrix * self.weights.T) / (
            self.probs_matrix @ self.weights
        )
        if self.Hij.isnan().any():
            raise ValueError("Has None")

    def m_step(self):
        self.dyads = (
            self.starts.min()
            + torch.argmax(self.grid_probs.T @ self.Hij, dim=0, keepdim=True).T
        )
        self.weights = self.Hij.sum(0, keepdim=True).T
        self.weights = self.weights / self.weights.sum() - self.reg_coef
        self.weights[self.weights < 0] = 0
        self.weights /= self.weights.sum()
        if (self.weights == 0).all() or self.weights.isnan().all():
            raise ValueError("all weights = 0. It seems reg_coef is too high")

    def delete_components(self):
        keep_alive_mask = (self.weights > 0).bool().reshape(-1, 1)
        self.weights = self.weights[keep_alive_mask].reshape(-1, 1)
        self.weights /= self.weights.sum()
        self.dyads = self.dyads[keep_alive_mask].reshape(-1, 1)
        self.probs_matrix = self.__create_probs_matrix(self.dyads)

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        self.starts = self.starts.to(device)
        self.stops = self.stops.to(device)
        self.errors = self.errors.to(device)
        self.dyads = self.dyads.to(device)
        self.weights = self.weights.to(device)
        if self.probs_matrix is not None:
            self.probs_matrix = self.probs_matrix.to(device)
        if self.Hij is not None:
            self.Hij = self.Hij.to(device)

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
        lh = (
            torch.log((self.probs_matrix * self.weights.T).sum(axis=0)).sum()
            - self.reg_coef * torch.log(self.weights).sum()
        )
        return lh

    def fit(self, starts, stops, dyads=None, weights=None):
        self.starts = self.__validate_cords(starts)
        self.stops = self.__validate_cords(stops)
        self.dyads = self.__validate_dyads(dyads)
        self.probs_matrix = self.__create_probs_matrix(self.dyads)
        self.grid_probs = self.__create_grid_probs()
        self.weights = self.__validate_weights(weights)
        self.e_step()

        nweights = []

        prev_lh = -100000
        prev_sliding_mean = -100000
        lh_loss = []
        sliding_mean_list = []
        self.df_list = []

        for i in range(self.max_iter):
            self.e_step()
            self.df_list.append(self.to_df())
            self.m_step()
            self.delete_components()

            nweights.append(len(self.weights))

            cur_lh = self.logLH()
            lh_loss.append(cur_lh.item())
            if i == 0:
                sliding_mean = cur_lh  
            else:
                sliding_mean = (
                    self.alpha * cur_lh + (1 - self.alpha) * prev_sliding_mean
                )

            sliding_mean_list.append(sliding_mean.item())
            prev_lh = cur_lh
            delta_sliding_mean = sliding_mean - prev_sliding_mean
            prev_sliding_mean = sliding_mean

            if delta_sliding_mean < self.tol:
                self.e_step()
                return {"lhloss": lh_loss, "sliding_mean": sliding_mean_list, 'success': True, "nweights": nweights}
        self.e_step()
        return {"lhloss": lh_loss, "sliding_mean": sliding_mean_list, 'success': False, "nweights": nweights}

    def to_df(self):
        df = pd.DataFrame({"start": self.starts.flatten().cpu(), "stop": self.stops.flatten().cpu()})
        df["dyads"] = self.dyads[self.Hij.argmax(1)].cpu()
        df["template|dyad"] = self.Hij.max(1)[0].cpu()
        return df


class StochasticEMMOdel(EMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def m_step(self):
        stochastic_res = self.__sample_multinomial_vectorized_torch()
        weighted_sum = torch.matmul(self.grid_probs.T, stochastic_res)
        max_indices = torch.argmax(weighted_sum, dim=0, keepdim=True).T
        self.dyads = self.starts.min() + max_indices
        self.weights = stochastic_res.sum(0, keepdim=True).T
        weights_sum = self.weights.sum()
        if weights_sum != 1:
            self.weights = self.weights / weights_sum - self.reg_coef
        else:
            self.weights = self.weights - self.reg_coef
        self.weights = torch.where(
            self.weights < 0, torch.tensor(0.0, device=self.device), self.weights
        )
        self.weights = self.weights / self.weights.sum()

    def __sample_multinomial_vectorized_torch(self):
        if torch.isnan(self.Hij).any():
            raise ValueError("Hij matrix contains nan")
        samples = torch.multinomial(self.Hij, num_samples=1).squeeze(-1)
        n_classes = self.Hij.shape[-1]
        stochastic_res = torch.zeros(
            (samples.shape[0], n_classes), device=self.device, dtype=torch.double
        )
        stochastic_res.scatter_(1, samples.unsqueeze(-1), 1.0)

        return stochastic_res
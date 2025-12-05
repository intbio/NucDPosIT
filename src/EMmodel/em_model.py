import torch
import copy


class EMModel:
    def __init__(
        self,
        starts,
        stops,
        errors,
        dyad_dist,
        max_iter=500,
        reg_coef=0,
        tol=0.0001,
        dyads=None,
        weights=None,
        device="cpu",
    ):
        self.device = device
        self.starts = self.__validate_cords(starts)
        self.stops = self.__validate_cords(stops)
        self.errors = errors.to(device)
        self.dyad_dist = dyad_dist
        self.max_iter = max_iter
        self.dyads = self.__validate_dyads(dyads)
        self.weights = self.__validate_weights(weights)
        self.reg_coef = reg_coef
        self.tol = tol
        self.probs_matrix = self.__create_probs_matrix(self.dyads)
        self.Hij = torch.eye(*self.probs_matrix.shape, device=device)
        self.grid_probs = self.__create_grid_probs()

    @staticmethod
    def reset(model):
        new_model = type(model)(
            model.starts,
            model.stops,
            model.errors,
            model.ndyads,
            model.max_iter,
            model.reg_coef,
            model.tol,
            None,
            None,
            model.device,
        )
        return new_model

    @property
    def nreads(self):
        return len(self.starts)

    @property
    def ndyads(self):
        return len(self.dyads)

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
            weights = torch.rand(
                self.ndyads, dtype=float, device=self.device
            ).reshape(-1, 1)
            weights /= weights.sum()
        return weights

    def __create_probs_matrix(self, dyads):
        left_ones = (
            torch.ones((self.nreads, len(dyads)), dtype=float, device=self.device)
            * self.starts
        ).T
        right_ones = (
            torch.ones((self.nreads, len(dyads)), dtype=float, device=self.device)
            * self.stops
        ).T
        L_index = self.__insert_probs_to_matrix(dyads - left_ones, self.errors)
        R_index = self.__insert_probs_to_matrix(right_ones - dyads, self.errors)
        return (L_index * R_index).T

    def __create_grid_probs(self):
        potential_dyads = torch.arange(
            self.starts.min(), self.stops.max() + 1, 1, device=self.device
        ).reshape(-1, 1)
        probs = torch.log(self.__create_probs_matrix(potential_dyads) + 1e-50)
        return probs

    def __insert_probs_to_matrix(self, idx_matrix, errors):
        valid_mask = ((idx_matrix >= 0) & (idx_matrix < len(errors))).bool()
        idx_matrix[valid_mask] = errors[idx_matrix[valid_mask].int()]
        idx_matrix[~valid_mask] = 0
        return idx_matrix

    def e_step(self):
        self.probs_matrix = self.__create_probs_matrix(self.dyads)
        self.Hij = (self.probs_matrix * self.weights.T) / (
            self.probs_matrix @ self.weights
        )

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

    def run(self):
        cold_iter = 10
        prev = 0
        losses = []

        for i in range(self.max_iter):
            prev_hij = self.Hij

            self.e_step()
            self.m_step()

            if self.Hij.shape != prev_hij.shape:
                continue

            loss = (self.Hij - prev_hij).abs().sum().tolist()
            losses.append(loss)

            # if loss < self.tol:
            #     break

        # self.__delete_components()
        return losses

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


class StochasticEMMOdel(EMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def m_step(self):
        stochastic_res = self.__sample_multinomial_vectorized_torch()
        self.dyads = (
            self.starts.min()
            + torch.argmax(self.grid_probs.T @ stochastic_res, dim=0, keepdim=True).T
        )
        
        self.weights = stochastic_res.sum(0, keepdim=True).T  # Сумма по reads для каждого дайда
        self.weights = self.weights / self.weights.sum() - self.reg_coef
        
        self.weights[self.weights < 0] = 0
        self.weights /= self.weights.sum()
        if (self.weights == 0).all():
            raise ValueError("all weights = 0. It seems reg_coef is too high")

    def __sample_multinomial_vectorized_torch(self):
        if self.Hij.isnan().any():
            raise ValueError("Hij matrix contains nan")
        samples = torch.multinomial(self.Hij, num_samples=1).squeeze(-1)
        n_classes = self.Hij.shape[-1]
        stochastic_res = torch.nn.functional.one_hot(
            samples, num_classes=n_classes
        ).float()
        return stochastic_res.double()

    def run(self):
        cold_iter = 0
        smoothed_change = 0
        alpha = 0.3  # коэффициент сглаживания
        history = []
        convergence_count = 0

        for i in range(self.max_iter):
            prev_hij = self.Hij

            if i < cold_iter:
                super().e_step()
                super().m_step()

            else:
                self.delete_components()
                self.e_step()
                self.m_step()

            if self.Hij.shape != prev_hij.shape:
                continue

            # Вычисляем изменение
            current_change = (self.Hij - prev_hij).abs().mean().item()

            # Экспоненциальное сглаживание
            smoothed_change = alpha * current_change + (1 - alpha) * smoothed_change
            history.append(smoothed_change)

            # Условия сходимости
            if i > cold_iter and smoothed_change < self.tol:
                # Дополнительная проверка: стабильность в течение нескольких итераций
                convergence_count += 1

                if convergence_count >= 5:
                    break
            else:
                convergence_count = 0

        return history
    
    
class ModelOptimizer:
    def __init__(
        self,
        model_class,
        errors,
        dyad_dist,
        max_model_iter=500,
        max_train_iter=20,
        max_successful_runs=3,
        nretries=5,
        device="cpu",
    ):
        self.model_class = model_class
        self.errors = errors
        self.device = device

        self.dyad_dist = dyad_dist
        self.max_model_iter = max_model_iter
        self.max_train_iter = max_train_iter
        self.max_successful_runs = max_successful_runs
        self.nretries = nretries

    def optimize_reg_coef(self, starts, stops, dyad_dist):
        successful_runs = 0
        low = 0
        high = 1
        best_model = None

        for i in range(self.max_train_iter):
            mid = (low + high) / 2
            new_model = self.model_class(
                starts,
                stops,
                self.errors,
                reg_coef=mid,
                dyad_dist=dyad_dist,
                max_iter=self.max_model_iter,
                device=self.device,
            )

            try:
                new_model.run()

            except Exception as error:
                high = mid

            else:
                low = mid
                successful_runs += 1

                if (
                    best_model is None
                    or (new_model.weights != 0).sum().item()
                    < (best_model.weights != 0).sum().item()
                ):
                    best_model = copy.deepcopy(new_model)

            if successful_runs >= self.max_successful_runs:
                break

        if best_model is None:
            raise ValueError("best model is None")
        return best_model

    def fit_data(self, starts, stops):
        success = False
        best_model = None

        try:
            best_model = self.optimize_reg_coef(starts, stops, self.dyad_dist)
            success = True
            print(f"Success with dyad_dist={self.dyad_dist}")
            return best_model

        except Exception as e:
            for i in range(self.nretries):
                new_dyad_dist = self.dyad_dist // 2 ** (i + 1)
                if new_dyad_dist == 0:
                    new_dyad_dist = 1            

                try:
                    best_model = self.optimize_reg_coef(
                        starts, stops, new_dyad_dist
                    )
                    success = True
                    print(f"Success on retry {i+1} with dyad_dist={new_dyad_dist}")
                    return best_model

                except Exception as retry_error:
                    print(f"Retry {i+1} failed: {retry_error}")
                    continue

        raise ValueError("empty model")

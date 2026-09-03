import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ErrorsProbs:
    def __init__(self, handler, lamb, *args, **kwargs):
        self.__errors = self.load_errors(handler, lamb)

    def __len__(self):
        return len(self.__errors)

    @property
    def errors(self):
        return self.__errors

    def load_errors(self, handler, lamb=None):
        if isinstance(handler, str):
            if lamb is None:
                raise ValueError("regularization coefficient lambda is None")
            errors_df = pd.read_csv(handler)
            q = errors_df.query("lamb == @lamb")
            if len(q) == 0:
                raise ValueError(f"regularization coefficient lambda={lamb} is not found in {handler}")
            if len(q) != 1:
                raise ValueError(f"several rows were found in handler {handler}")
            list_str = q['result_x'].iloc[0]
            errors = np.fromstring(list_str.strip('[]'), sep=' ')
        elif isinstance(handler, list):
            errors = np.array(handler)
        else:
            raise TypeError("handler must be either a string (file path) or a list of errors")
        
        sum_errors = errors.sum()
        if sum_errors == 0:
            raise ValueError("Sum of errors is zero, cannot normalize")
        
        errors = errors / sum_errors
        return errors

    def __getitem__(self, items):
        if not isinstance(items, np.ndarray):
            items = np.array(items)
        res = np.zeros_like(items, dtype=float)
        mask = ((items >= 0) & (items < len(self.errors)))
        res[mask] = self.errors[items[mask]]
        return res

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.errors)

    def len(self):
        return len(self.__errors)

    def generate_random_len(self, size):
        lengths = len(self.errors)
        random_lens = np.random.choice(lengths, size, p=self.errors)
        return random_lens

    def generate_dyad_probs(self, size):
        left_cords = -self.generate_random_len(size)
        right_cords = self.generate_random_len(size)
        length = right_cords - left_cords + 1
        df = pd.DataFrame(
            [left_cords, right_cords, length], index=["start", "end", "len"]
        ).T
        grouped_df = df.groupby("len")
        histograms = dict()
        for name, group in grouped_df:
            centers = (group["start"] + group["end"]) // 2
            delta_dyad = centers
            hist = np.histogram(
                delta_dyad.to_numpy(), bins=np.arange(-201, 201), density=True
            )
            histograms[name] = hist[0]
        return histograms

    def get_template_prob(self, dyad, start, stop):
        l1 = dyad - start + 1
        if l1 < len(self.errors):
            p1 = self.errors[l1]
        else:
            p1 = 0
        l2 = stop - dyad + 1
        if l2 < len(self.errors):
            p2 = self.errors[l2]
        else:
            p2 = 0
        return p1 * p2

    def fit_model_template(self, size=100000):
        errors = self.errors
        vals = np.arange(len(errors))
        random_lefts = -np.random.choice(vals, size=size, p=errors)
        random_rights = np.random.choice(vals, size=size, p=errors)
        all_values = np.concatenate([random_lefts, random_rights + 1])
        min_val = int(min(all_values)) - 50
        max_val = int(max(all_values)) + 50
        diff_array = np.zeros(max_val - min_val + 2, dtype=np.int64)
        left_indices = random_lefts - min_val
        right_indices = random_rights + 1 - min_val
        left_counts = np.bincount(left_indices.astype(int), minlength=len(diff_array))
        right_counts = np.bincount(right_indices.astype(int), minlength=len(diff_array))
        diff_array = left_counts - right_counts
        result = np.cumsum(diff_array) / size
        return result
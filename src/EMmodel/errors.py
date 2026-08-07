import matplotlib.pyplot as plt
import numpy as np


class ErrorsProbs:
    def __init__(self, handler, *args, **kwargs):
        self.__errors = self.load_errors(handler)

    def __len__(self):
        return len(self.__errors)

    @property
    def errors(self):
        return self.__errors

    def load_errors(self, handler):
        if isinstance(handler, str):
            errors = np.loadtxt(handler, delimiter=",")
        if isinstance(handler, list):
            errors = np.array(handler)
        errors /= errors.sum()
        return errors

    def __getitem__(self, items):
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
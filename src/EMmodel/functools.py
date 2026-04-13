import numpy as np
import copy
from collections import Counter
import numpy as np
from tqdm.auto import tqdm


def make_occupancy(starts, stops):
    min_value = starts.min()
    max_value = stops.max()
    x = np.arange(min_value, max_value + 1)
    occ = np.zeros_like(x)
    for start, stop in zip(starts, stops):
        occ[start - min_value : stop - min_value + 1] += 1
    return x, occ


def fit_model_template(errors, size=100000):
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




import numpy as np
import copy
from collections import Counter
import numpy as np
from tqdm.auto import tqdm



class NucleosomeContainer:
    pass

class ErrorsProbs:
    pass


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


def build_occupancy(container: NucleosomeContainer, min_pos=None, max_pos=None, pad=0):

    min_pos = (
        min(nuc.left for nuc in container._nucleosomes) if not min_pos else min_pos
    ) - pad
    max_pos = (
        max(nuc.right for nuc in container._nucleosomes) if not max_pos else max_pos
    ) + pad

    assert min_pos < max_pos, "positioin error"

    # Создаем массив покрытия
    occupancy = {pos: 0 for pos in range(min_pos, max_pos + 1)}

    # Для каждой нуклеосомы увеличиваем счетчик на её позициях
    for nuc in container._nucleosomes:
        for pos in range(nuc.left, nuc.right + 1):
            occupancy[pos] += nuc.count

    return occupancy


def build_dyads(container: NucleosomeContainer, min_pos=None, max_pos=None, pad=0):
    min_pos = (
        min(nuc.left for nuc in container._nucleosomes) if not min_pos else min_pos
    ) - pad
    max_pos = (
        max(nuc.right for nuc in container._nucleosomes) if not max_pos else max_pos
    ) + pad

    assert min_pos < max_pos, "positioin error"

    occupancy = {pos: 0 for pos in range(min_pos, max_pos + 1)}
    for nuc in container.iter_nucleosomes():
        occupancy[nuc.dyad] += nuc.count
    return occupancy

def build_template_occupancy(container: NucleosomeContainer, errors: ErrorsProbs, min_pos=None, max_pos=None, pad=0):
    assert pad >= 0
    template = errors.fit_model_template()
    pad += len(template)
    
    min_pos = (
        min(nuc.left for nuc in container._nucleosomes) if not min_pos else min_pos
    ) - pad
    max_pos = (
        max(nuc.right for nuc in container._nucleosomes) if not max_pos else max_pos
    ) + pad

    assert min_pos < max_pos, "positioin error"
    occupancy = {pos: 0 for pos in range(min_pos, max_pos + 1)}
    for nuc in container.iter_nucleosomes():
        for i in range(len(template)):
            occupancy[nuc.dyad - len(template) // 2 + i] += template[i] * nuc.count
    for i in range(len(template)):
        occupancy.pop(min_pos + i)
        occupancy.pop(max_pos - i)
    return occupancy
    




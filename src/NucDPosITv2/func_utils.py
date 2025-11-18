from nuc_template_constants import ExoNucTemplate


from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
import scipy as spy


def make_model_occupancy(model, left_cords, right_cords, pad=400):
    offset = left_cords.min()
    occupancy = np.zeros(right_cords.max() - offset)
    for b, e in zip(left_cords - offset, right_cords - offset):
        occupancy[b:e] += 1
    occupancy = np.pad(occupancy, pad)
    
    counter = Counter()
    for pos, wei in zip(model.positions - offset, model.gij.sum(0)):
        counter[int(pos)] += float(wei)
    
    x = np.zeros_like(occupancy)
    for p, w in counter.items():
        try:
            x[p+pad - 200 : p +pad+ 200] += ExoNucTemplate().nuc_template * w
        except Exception:
            print(p, w)
    return x


def make_dyad_pos(model, offset):
    counter = Counter()
    for pos, wei in zip(model.positions - offset, model.gij.sum(0)):
        counter[int(pos)] += float(wei)
    return counter

def count_loss(model):
    neg_LH = -torch.log(model.gij.float() @ model.weights.float()).sum() + model.tau * torch.log(model.weights[model.weights != 0]).sum()
    neg_LH = torch.nan if neg_LH == 0.0 else neg_LH
    return neg_LH


def make_position_matrix(starts, stops, grouped_var=None, start=None, stop=None):
    return _make_position_matrix(starts, stops, grouped_var, start, stop)


def _make_position_matrix(starts, stops, grouped_var=None, start=None, stop=None):
    grouped_var = np.ones_like(starts) if grouped_var is None else grouped_var
    min_start = start if start is not None else starts.min()
    stop = stop if stop is not None else stops.max()
    imshow = np.zeros((len(starts), stop - min_start))
    for i, (cur_start, cur_stop) in enumerate(zip(starts, stops)):
        imshow[i, cur_start - min_start : cur_stop - min_start] = grouped_var[i]
    return imshow


def make_window_df(model, batch):
    if model is None:
        return None
    prob_matrix = model.gij
    window_data = pd.DataFrame(batch, columns=['start', 'stop'])
    window_data['mid'] = (window_data.start + window_data.stop) / 2
    nuc_indxex = torch.argmax(model.gij, dim=1)
    window_data["dyad"] = model.positions[nuc_indxex]
    # window_data["readLH"] = torch.max(prob_matrix, dim=1)[0]
    window_data['nuc_prob'] = model.weights[nuc_indxex]
    return window_data


def fit_model(model, left_cords, right_cords, max_iter=200, eps=1e-6):
    model.init_fields(left_cords, right_cords)
    n_iter = 0
    weights = model.weights
    _, new_weights = model(left_cords, right_cords)
    loss = torch.abs(weights - new_weights).sum()
    while loss > eps and n_iter < max_iter:
        n_iter += 1
        _, new_weights = model(left_cords, right_cords)
        loss = torch.abs(weights - new_weights).sum()
        weights = new_weights
    return model
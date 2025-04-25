import numpy as np
import plotly.express as px
import math


# @jit
def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


# @jit
def jit_get_left_prob(left_cord, dyad_pos, fit_res):
    i = dyad_pos - left_cord - 23
    if i < 0 or i >= len(fit_res):
        return 0
    return fit_res[i]


# @jit
def jit_get_right_prob(right_cord, dyad_pos, fit_res):
    i = right_cord - dyad_pos - 23
    if i < 0 or i >= len(fit_res):
        return 0
    return fit_res[i]


def make_position_matrix(batch, grouped_var=None, start=None, stop=None):
    starts, stops = batch[:, 0], batch[:, 1]
    return _make_position_matrix(starts, stops, grouped_var, start, stop)


def _make_position_matrix(starts, stops, grouped_var=None, start=None, stop=None):
    grouped_var = np.ones_like(starts) if grouped_var is None else grouped_var
    min_start = start if start else starts.min()
    stop = stop if stop else stops.max()
    imshow = np.zeros((starts.size, stop - min_start))
    for i, (cur_start, cur_stop) in enumerate(zip(starts, stops)):
        imshow[i, cur_start - min_start : cur_stop - min_start] = grouped_var[i]
    return imshow


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    for _ in range(math.ceil(len(iterable) / n)):
        batch = np.array(list((itertools.islice(it, n))))
        yield batch
        
        
def __custom_cv(X, n=5):
    for i in range(n):
        yield np.arange(X.shape[0]), np.arange(X.shape[0])
        

def plot_view(viewer, nuc_template):
    x, coverage = viewer.make_coverage()
    model_coverage = viewer.model_occ(nuc_template)[1]
            # model_std = self.model_std(std_template)[1]
            # ax.errorbar(x, model_coverage, yerr=model_std, label='model')
    fig = px.line( x = x , 
                  y = coverage)
    fig.add_trace(px.line( x = x , 
                           y = model_coverage).data[0])
    fig['data'][0]['line']['color']='orange'
    fig['data'][1]['line']['color']='blue'

    fig.add_trace(px.bar(x = viewer.stat_df.dyad.to_numpy(), y = viewer.stat_df.height.to_numpy(), width=1000).data[0])
    fig.show()
import numpy as np
from numba import jit

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


@jit
def jit_get_left_prob(left_cord, dyad_pos, fit_res):
    i = dyad_pos - left_cord - 23
    if i < 0 or i >= len(fit_res):
        return 0
    return fit_res[i]


@jit
def jit_get_right_prob(right_cord, dyad_pos, fit_res):
    i = right_cord - dyad_pos - 23
    if i < 0 or i >= len(fit_res):
        return 0
    return fit_res[i]

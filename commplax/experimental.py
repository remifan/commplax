import numpy as np
from jax import jit, vmap, numpy as jnp
from numpy.polynomial.polyutils import RankWarning


def polyfit(x, y, deg, rcond=None):
    '''
    Reference:
    https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/polynomial/polyutils.py#L641
    [BUG]:
    1. jit on GPU somehow demands insanely massive memeory
    2. inaccurate by using default float32 based rcond, use np.float64 as workaround
    '''
    x = jnp.atleast_1d(x) + 0.0
    y = jnp.atleast_1d(y) + 0.0
    order = np.asarray(deg) + 1
    if rcond is None:
        # standard rcond is buggy in jax, might be related to float32
        # rcond = x.shape[0] * jnp.finfo(x.dtype).eps
        rcond = x.shape[0] * jnp.finfo(np.float64).eps

    c, [resids, rank, s, rcond] = jit(_polyfit, static_argnums=(2, 3), backend='cpu')(x, y, order, rcond)
    # if rank != order and not full:
    #     msg = "The fit may be poorly conditioned, rank=%d, oder=%d" % (rank, order)
    #     warnings.warn(msg, RankWarning, stacklevel=2)

    return c


def _polyfit(x, y, order, rcond):
    lhs = jnp.vander(x, order).T
    rhs = y.T
    scl = jnp.sqrt(jnp.square(lhs).sum(1))
    scl = jnp.where(scl == 0, 1, scl)
    c, resids, rank, s = jnp.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T
    return c, [resids, rank, s, rcond]


def polyfitval(x, y, deg):
    return vmap(lambda x, y: jnp.polyval(polyfit(x, y, deg), x), in_axes=-1, out_axes=-1)(x, y)



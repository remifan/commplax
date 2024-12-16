import numpy as np
import dataclasses as dc
from jax import numpy as jnp, lax
from typing import Callable
from scipy.special import iv
from scipy.signal import firwin
from typing import NamedTuple 
import equinox as eqx
from equinox import field
from jaxtyping import Array, Float, Int, PyTree
from commplax import util as cu
from commplax.buffer import SyncFIFO as FIFO


class _Resampler(NamedTuple):
    init: Callable
    apply: Callable


class VarRateResampler(eqx.Module):
    state: PyTree
    fifo: FIFO
    acc_phase: Array
    op: _Resampler = field(static=True)

    def __init__(self, ratio=2, state=None, acc_phase=None, op=None, fifo=None, dtype=None):
        dtype = cu.default_complexing_dtype() if dtype is None else dtype
        self.op = var_rate_poly_resampler(ratio) if op is None else op
        self.state = self.op.init(10, dtype=dtype) if state is None else state
        self.fifo = FIFO(self.state[0].shape[1], dtype=dtype) if fifo is None else fifo
        self.acc_phase = jnp.array(0.) if acc_phase is None else acc_phase

    def __call__(self, x):
        fifo, _ = self.fifo(x)
        state, y = self.op.apply(self.state, fifo.state[0], self.acc_phase)
        rsplr = dc.replace(self, fifo=fifo, state=state)
        return rsplr, y


def var_rate_poly_resampler(ratio, Q=128, interp_method='nearest neighbor'):
    '''
    A arbitrary-ratio resampler uses a polyphase filter bank for interpolation between
      available input sample points.
    args:
      ratio: f_out / f_in (p / q)
    notes:
    The general state flow refers to [1] 
    references:
    [1] Multirate Signal Processing for Communication Systems, Harris. (Figure 7.32, pp179)
    '''
    assert interp_method.lower() in ['nearest neighbor', 'two neighbor']
    ctype = cu.default_complexing_dtype()
    ftype = cu.default_floating_dtype()

    δ = Q / ratio
    N = int(np.ceil(ratio)) # TODO addressing roundup for integer ratio under certain ppm
    Nzeros = jnp.zeros(N)

    def init(subfir_halflen=10, h=None, window=('kaiser', 5.0), dtype=None):
        if h is None:
            # design kaiser filter following Scipy's implementation
            max_rate = max(Q, Q / ratio)
            f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
            half_len = subfir_halflen * int(max_rate)
            h = firwin(2 * half_len + 1, f_c, window=window)
        dtype = ctype if dtype is None else dtype
        hs = jnp.fliplr(_poly_decomp(h, Q)).astype(dtype) * Q  # no pads needed in decomposition
        acc = jnp.array(0, dtype=ftype)
        state = hs, acc
        return state

    def apply(state, u, ε):
        hs, acc = state
        assert u.ndim == 1 and u.shape[0] == hs.shape[1]

        Δ = δ + ε # ε: user input sampling phase offset
        # iteratively predict the sub-filter index; the search process is bounded (N) by stuffing
        #   a dummy sub-filter to ensure static output shape, otherwise resorting to jax.where_loop
        #   would be at least 10 times slower for unknown reason.
        acc, ind = lax.scan(
            lambda a, _: lax.cond(a < Q, lambda *_: (a+Δ, a), lambda *_: (a, -1.)),
            acc, Nzeros, unroll=True)
        # acc = jnp.fmod(acc, Q) # described in [1], but it does not apply to downsampling
        acc = acc - Q            # where acc may be multiples of Q
        _hs = jnp.pad(hs, [[0,1],[0,0]], constant_values=jnp.nan)
        # apply the sub-filters
        match interp_method.lower():
            case 'nearest neighbor':
                ind = jnp.clip(jnp.round(ind).astype(int), max=Q-1)
                # ind = jnp.floor(ind).astype(int)
                v = jnp.dot(_hs[ind], u)
            case 'two neighbor':
                floor_ind = jnp.floor(ind).astype(int)
                ceil_ind = jnp.clip(floor_ind + 1, max=Q-1)
                delta = ind - floor_ind
                v = jnp.dot(
                    jnp.concatenate([_hs[floor_ind], _hs[ceil_ind]], axis=-1),
                    jnp.concatenate([u * (1 - delta), u * delta], axis=-1))

        state = hs, acc
        return state, v

    return _Resampler(init, apply)


def _poly_decomp(h, Q):
    h = jnp.atleast_1d(h)
    if not h.ndim == 1:
        raise ValueError("polyphase filter ndim must be 1")
    L = h.shape[0]
    hs = jnp.pad(h, [0, Q - L % Q]).reshape((Q, -1), order='F')
    return hs

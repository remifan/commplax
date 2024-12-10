import numpy as np
from jax import numpy as jnp, lax
from scipy.special import iv
from scipy.signal import firwin
from commplax import util as cu


def poly_resampler(ratio, Q=128):
    '''
    A arbitrary-ratio resampler uses a polyphase filter bank for interpolation between
      available input sample points.
    The general state flow is based on the architecture fig. 7.32 in [1]
    references:
    [1] Multirate Signal Processing for Communication Systems, Harris.
    '''
    ctype = cu.default_complexing_dtype()
    ftype = cu.default_floating_dtype()

    d = Q / ratio
    N = int(np.ceil(ratio)) # TODO addressing roundup for integer ratio

    def init(h=None, window=('kaiser', 5.0)):
        if h is None:
            # design kaiser filter following Scipy's implementation
            max_rate = Q # max(Q, q) for p, q specification
            f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
            half_len = 10 * max_rate  # reasonable cutoff for sinc-like function
            h = firwin(2 * half_len + 1, f_c, window=window)
        hs = jnp.fliplr(_poly_decomp(h, Q)) * Q
        acc = jnp.array(0, dtype=ftype)
        return hs, acc 

    def apply(state, u):
        hs, acc = state
        assert u.ndim == 1 and u.shape[0] == hs.shape[1]

        # iteratively predict the sub-filter index
        I = jnp.full(Q, False)
        I, acc = lax.while_loop(
            lambda x: x[1] < Q,
            lambda x: (x[0].at[jnp.floor(x[1]).astype(int)].set(True), x[1]+d),
            (I, acc))
        acc = jnp.remainder(acc, Q)
        # stuff a dummy sub-filter to ensure static output shape
        #   at the cost of additional one sub-filtering
        _hs = jnp.pad(hs, [[0,1],[0,0]], constant_values=jnp.nan)
        # jax.debug.print("{a}", a=I)
        ind = jnp.where(I, size=N, fill_value=-1)[0]
        # apply the sub-filters
        v = jnp.dot(_hs[ind], u) # could be empty

        state = hs, acc
        return state, v

    return init, apply


def _poly_decomp(h, Q):
    h = jnp.atleast_1d(h)
    if not h.ndim == 1:
        raise ValueError("polyphase filter ndim must be 1")
    L = h.shape[0]
    hs = jnp.pad(h, [0, Q - L % Q]).reshape((Q, -1), order='F')
    return hs
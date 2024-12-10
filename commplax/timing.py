import numpy as np
import jax
from jax import numpy as jnp, lax
from commplax.util import default_complexing_dtype, default_floating_dtype

def symbol_timing_sync():
    ''' operate at 2 samples per symbol
        adapted from the Matlab codes in [1](pp.493)
    References:
      [1] Digital Communications_ A Discrete-Time Approach -- Rice, Michael
    '''
    # TODO add spec for these parameters
    K1 = -2.46e-3
    K2 = -8.2e-6

    b = 1/2 * jnp.flip(jnp.array(
        [[ 1, -1, -1,  1],
         [-1,  3, -1, -1],
         [ 0,  0,  2,  0]], dtype=default_complexing_dtype()),
        axis=1)

    def init(dtype=None):
        dtype = default_complexing_dtype() if dtype is None else dtype
        η_next = 0.
        μ_next = 0.
        strobe = 0
        B = jnp.zeros(2, dtype=dtype) # TED buffer
        vi = 0.
        state = μ_next, η_next, strobe, vi, B
        return state

    def apply(state, x):
        μ_next, η_next, strobe, vi, B = state

        μ = μ_next
        η = η_next
        m = jnp.power(μ, jnp.arange(2,-1,-1))
        xI = jnp.dot(jnp.dot(b, x), m)

        y = jnp.where(strobe//2 == 1, xI, jnp.nan)

        e  = jnp.where(
            strobe == 2,
            B[0].real * (B[1].real - xI.real) + B[0].imag * (B[1].imag - xI.imag),
            0.,
            )

        vp = K1 * e
        vi = vi + K2 * e
        v = vp + vi
        W = 1 / 2 + v

        B = lax.cond(
            (strobe == 1) | (strobe == 2),
            lambda *_: jnp.array([xI, B[0]], dtype=B.dtype),
            lambda *_: lax.cond(
                strobe == 0,
                lambda *_: B,
                lambda *_: jnp.array([xI, 0.], dtype=B.dtype),
            )
        )
        η_next = η - W
        η_next, strobe, μ_next = lax.cond(
            η_next < 0,
            lambda *_: (η_next+1, 2+strobe//2, η/W),
            lambda *_: (η_next,   0+strobe//2, μ),
        )

        state = μ_next, η_next, strobe, vi, B
        return state, (y, e, μ)

    return init, apply


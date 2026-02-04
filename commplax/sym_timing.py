# Copyright 2026 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
from jax import lax, numpy as jnp
import numpy as np
import equinox as eqx
from equinox import field
from jaxtyping import Array, Float, Int, PyTree
from typing import Any, TypeVar, Callable, Optional, Tuple, Union
import dataclasses as dc
from commplax.jax_util import default_complexing_dtype, default_floating_dtype


InitFn = Callable
ApplyFn = Callable

@dc.dataclass
class SymbolTimingSync():
    init: InitFn = None
    apply: ApplyFn = None

    def __iter__(self):
        return iter((self.init, self.apply))

class SymSync(eqx.Module):
    fifo: Array
    state: PyTree
    af: PyTree = field(static=True)

    def __init__(self, af=None, state=None, fifo=None, dtype=None, af_kwds={}):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.af = symbol_timing_sync(**af_kwds) if af is None else af
        self.state = self.af.init() if state is None else state
        self.fifo = jnp.zeros(4, dtype=dtype) if fifo is None else fifo

    def __call__(self, input):
        fifo = jnp.roll(self.fifo, -1, axis=0).at[-1:].set(input)
        state, out = self.af.apply(self.state, fifo)
        ss = dc.replace(self, fifo=fifo, state=state)
        return ss, out[0]


def symbol_timing_sync():
    ''' operate at 2 samples per symbol
        adapted from the Matlab codes in [1]
    References:
      [1] Digital Communications A Discrete-Time Approach -- Rice, Michael (pp.493)
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

    return SymbolTimingSync(init, apply)


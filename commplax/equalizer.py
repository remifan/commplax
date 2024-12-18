import dataclasses as dc
import numpy as np
from jax import lax, numpy as jnp
import equinox as eqx
from equinox import field
from commplax import adaptive_filter as _af
from commplax.util import default_complexing_dtype, default_floating_dtype, astuple
from functools import lru_cache
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any, TypeVar
from numpy.typing import ArrayLike


@lru_cache
def rat_poly_inp_phases(up: int, down: int):
    '''
    see: https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_upfirdn_apply.pyx#L458
    '''
    _gcd = int(np.gcd(up, down))
    up //= _gcd
    down //= _gcd
    t = 0 # filter phase
    x_i = 0 # input index
    x_I = []
    while x_i < down:
        x_I.append(x_i)
        t += down
        x_i += t // up
        t = t % up
    return x_I


@lru_cache
def rat_poly_h_phases(h_len:int, up:int, flipped:bool=True):
    """Store coefficients indices in a transposed, flipped arrangement.
    see: https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_upfirdn.py#L46
    """
    h_padlen = -h_len % up
    ind = jnp.pad(jnp.arange(h_len), (0, h_padlen), constant_values=h_len) #index=h_len is out of bound
    ind = ind.reshape(
        (up, ind.shape[-1]//up), order='F'
        )[...,::1-2*int(flipped)]#.reshape((-1,))
    return ind


class MIMOCell(eqx.Module):
    fifo: Array
    state: Array
    inner_i: Array
    outer_i: Array
    in_phase: Array
    h_phase: Array
    up: int = field(static=True)
    down: int = field(static=True)
    af: PyTree = field(static=True)
    frozen: bool = field(static=True)

    def __init__(
        self,
        num_taps: int = 15,
        dims: int = 1,
        dtype=None,
        up: int = 1,
        down: int = 1,
        af: PyTree = None,
        state: Array = None,
        fifo: Array = None,
        inner_i: Array = jnp.array(0),
        outer_i: Array = jnp.array(0),
        in_phase: Array = None,
        h_phase: Array = None,
        frozen: bool = False,
    ):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.up = up
        self.down = down
        self.af = _af.lms() if af is None else af
        self.state = self.af.init(taps=num_taps, dims=dims, nspike=up) if state is None else state
        self.inner_i = jnp.asarray(inner_i) if inner_i is None else inner_i
        self.outer_i = jnp.asarray(outer_i) if outer_i is None else outer_i
        self.in_phase = jnp.asarray(rat_poly_inp_phases(up, down)) if in_phase is None else in_phase
        self.h_phase = jnp.asarray(rat_poly_h_phases(num_taps, up)) if h_phase is None else h_phase
        self.fifo = jnp.zeros((self.h_phase.shape[-1], dims), dtype=dtype) if fifo is None else fifo
        self.frozen = frozen

    def __call__(self, input: PyTree):
        x, *args = astuple(input)
        fifo = jnp.roll(self.fifo, -1, axis=0).at[-1:].set(x)
        # fetch the filter coeff with correct phase & form new state
        h_i = self.h_phase[self.inner_i % self.up]
        # enable flag of valid input
        valid_input_phase = jnp.any(self.outer_i % self.down == self.in_phase)
        state_i = (self.state[0].at[..., h_i].get(mode='fill', fill_value=0.),) + self.state[1:]
        output, inner_i = lax.cond(
            valid_input_phase,
            lambda *_: (self.af.apply(state_i, fifo), self.inner_i+1),
            lambda *_: (jnp.zeros_like(x), self.inner_i),
            )
        state_i = lax.cond(
            valid_input_phase & (not self.frozen),
            lambda *_: self.af.update(self.inner_i, state_i, (fifo, *args))[0],
            lambda *_: state_i,
            )
        state = (self.state[0].at[..., h_i].set(state_i[0]),) + state_i[1:]
        cell = dc.replace(self, fifo=fifo, state=state, inner_i=inner_i, outer_i=self.outer_i+1)
        return cell, output

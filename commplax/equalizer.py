import dataclasses as dc
import numpy as np
from jax import lax, numpy as jnp, debug as jdbg
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
        w0=None,
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
        self.state = self.af.init(w0=w0, taps=num_taps, dims=dims, dtype=dtype, nspike=up) if state is None else state
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
        i_w = self.h_phase[self.inner_i % self.up]
        # enable flag of valid input
        valid_input_phase = jnp.any(self.outer_i % self.down == self.in_phase)
        w_i = self.state[0].at[..., i_w].get(mode='fill', fill_value=0.)
        state_i = (w_i,) + self.state[1:]

        # apply the AF on the input
        dummy_out = jnp.zeros_like(x)
        output, inner_i = lax.cond(
            valid_input_phase,
            lambda *_: (self.af.apply(state_i, fifo), self.inner_i+1),
            lambda *_: (dummy_out, self.inner_i),
            )
        # update the AF states
        state_i = lax.cond(
            valid_input_phase & (not self.frozen),
            lambda *_: self.af.update(self.inner_i, state_i, (fifo, *args))[0],
            lambda *_: state_i,
            )

        w_i = self.state[0].at[..., i_w].set(state_i[0])
        state = (w_i,) + state_i[1:]
        cell = dc.replace(self, fifo=fifo, state=state, inner_i=inner_i, outer_i=self.outer_i+1)
        return cell, output


#TODO tap profiler to track clock drift


class FOE(eqx.Module):
    fo: float
    i: int
    t: int
    state: PyTree
    af: PyTree = field(static=True)
    uar: float = field(static=True)
    mode: str = field(static=True)

    def __init__(self, fo=0.0, uar=1.0, af=None, i=0, t=0, mode="feedforward", state=None, af_kwds={}):
        self.i = jnp.asarray(i)
        self.t = jnp.asarray(t)
        self.af = _af.foe_YanW(**af_kwds) if af is None else af
        self.fo = jnp.asarray(fo)
        self.uar = uar * 1.0
        fo4init = fo if mode == "feedforward" else 0.
        self.state = self.af.init(fo4init) if state is None else state
        self.mode = mode

    def __call__(self, input):
        if self.mode == "feedforward":
            foe = self.update(input)[0]
            foe, output = foe.apply(input)
        else:
            foe, output = foe.apply(input)
            foe = foe.update(output)[0]
        return foe, output

    def update(self, input):
        state, out = self.af.update(self.i, self.state, input)
        fo = self.fo + out[0] if self.mode == "feedback" else out[0]
        foe = dc.replace(self, fo=fo, state=state, i=self.i+1)
        return foe, None

    def apply(self, input):
        T = self.t + jnp.arange(input.shape[0])
        fo = self.fo * self.uar
        output = input * jnp.exp(-1j * fo * T)
        foe = dc.replace(self, t=T[-1]+1)
        return foe, output


class CPR(eqx.Module):
    i: int
    state: PyTree
    af: PyTree = field(static=True)
    mode: str = field(static=True)

    def __init__(self, af=None, dims=1, i=0, mode="feedforward", state=None, af_kwds={}):
        self.i = jnp.asarray(i)
        self.af = _af.cpane_ekf(**af_kwds) if af is None else af
        self.state = self.af.init(dims=dims) if state is None else state
        self.mode = mode

    def __call__(self, input):
        if self.mode == "feedforward":
            cpr = self.update(input)[0]
            cpr, output = cpr.apply(input)
        else:
            cpr, output = cpr.apply(input)
            cpr = cpr.update(output)[0]
        return cpr, output

    def update(self, input):
        state, _ = self.af.update(self.i, self.state, input)
        cpr = dc.replace(self, state=state, i=self.i+1)
        return cpr, None

    def apply(self, input):
        output = self.af.apply(self.state, input)
        return self, output

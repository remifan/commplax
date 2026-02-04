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


import dataclasses as dc
import numpy as np
from jax import lax, numpy as jnp, debug as jdbg
import equinox as eqx
from equinox import field
from commplax import adaptive_filter as _af
from commplax.jax_util import default_complexing_dtype, default_floating_dtype, astuple
from commplax._deprecated.cxopt import make_schedule, Union, Schedule
from functools import lru_cache
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any, TypeVar
from numpy.typing import ArrayLike


#@lru_cache  #lru_cache may conflict with jax.jit(jax.pmap)
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


#@lru_cache  #lru_cache may conflict with jax.jit(jax.pmap)
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


def af_update_flag(umode, train, i):
    cond0 = umode(i) > 0 # not frozen
    cond1 = train(i) # training
    cond2 = ~train(i) & (umode(i) == 1) # decison
    return cond0 & (cond1 | cond2)


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
    update_mode: Union[int, Schedule] = field(static=True)

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
        update_mode: Union[int, Schedule] = 1,
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
        self.update_mode = make_schedule(update_mode)

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
            valid_input_phase & af_update_flag(self.update_mode, self.af.update.train, inner_i),
            lambda *_: self.af.update(self.inner_i, state_i, (fifo, *args))[0],
            lambda *_: state_i,
            )

        w_i = self.state[0].at[..., i_w].set(state_i[0])
        state = (w_i,) + state_i[1:]
        cell = dc.replace(self, fifo=fifo, state=state, inner_i=inner_i, outer_i=self.outer_i+1)
        return cell, output


#TODO tap profiler to track clock drift


class FOE(eqx.Module):
    fo: Float
    i: Int
    t: Int
    state: PyTree
    af: PyTree = field(static=True)
    uar: float = field(static=True)
    mode: str = field(static=True)

    def __init__(self, fo=0.0, uar=1.0, af=None, i=0, t=0, mode="feedforward", state=None, af_kwds={}):
        self.i = jnp.asarray(i)
        self.t = jnp.asarray(t)
        self.af = _af.foe_YanW_ekf(**af_kwds) if af is None else af
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
        return foe, input

    def apply(self, input):
        T = self.t + jnp.arange(input.shape[0])
        fo = self.fo * self.uar
        output = input * jnp.exp(-1j * fo * T)
        foe = dc.replace(self, t=T[-1]+1)
        return foe, output


class CPR(eqx.Module):
    """Single-dimensional Carrier Phase Recovery module (symbol-by-symbol).

    Uses 4th-power PLL by default for robust phase tracking.
    For ensembles (e.g., dual-pol), use eqx.filter_vmap.

    Example:
        @eqx.filter_vmap
        def make_cpr_ensemble(_):
            return CPR(af=af.cpr_4thpower_pll(mu=0.01))
        cpr = make_cpr_ensemble(jnp.arange(2))

        @eqx.filter_vmap
        def cpr_step(cpr, x):
            return cpr(x)

        # Use with scan_with for symbol-by-symbol processing
        cpr, out = mod.scan_with(cpr_step)(cpr, signal)  # signal: (N, 2)
    """
    phase: Float
    i: Int
    af: PyTree = field(static=True)

    def __init__(self, phase=0.0, af=None, i=0):
        self.i = jnp.asarray(i)
        self.phase = jnp.asarray(phase)
        self.af = _af.cpr_4thpower_pll() if af is None else af

    def __call__(self, y):
        """Process single symbol: update phase and apply correction."""
        new_phase, _ = self.af.update(self.i, self.phase, y)
        y_corrected = self.af.apply(new_phase, y)
        cpr = dc.replace(self, phase=new_phase, i=self.i + 1)
        return cpr, y_corrected

    def update(self, y):
        """Update phase estimate from single symbol."""
        new_phase, aux = self.af.update(self.i, self.phase, y)
        cpr = dc.replace(self, phase=new_phase, i=self.i + 1)
        return cpr, y

    def apply(self, y):
        """Apply current phase correction to single symbol."""
        y_corrected = self.af.apply(self.phase, y)
        return self, y_corrected


def align_phase(signal, const, n_samples=1000):
    """Resolve phase ambiguity by testing candidate rotations.

    Tests 4 candidate phases (0°, 90°, 180°, 270°) and picks the one
    with minimum total distance to constellation points.

    Args:
        signal: Complex signal array, shape (..., N) or (N,)
        const: Reference constellation (e.g., sym_map.const('16QAM', norm=True))
        n_samples: Number of samples to use for alignment (default: 1000)

    Returns:
        (aligned_signal, best_phase): Rotated signal and the phase used

    Example:
        # After CPR (which has π/2 ambiguity)
        aligned, phase = align_phase(cpr_out, sym_map.const('16QAM', norm=True))
    """
    # Use subset for efficiency
    sample = signal.ravel()[:n_samples]

    # 4 candidates for QAM (90° symmetry)
    candidates = jnp.array([0., jnp.pi/2, jnp.pi, 3*jnp.pi/2])

    # Rotate by each candidate
    rotated = sample[None, :] * jnp.exp(-1j * candidates[:, None])

    # Distance to nearest constellation point for each candidate
    dist = jnp.min(jnp.abs(const[:, None, None] - rotated[None, :, :]), axis=0)
    total_dist = jnp.sum(dist**2, axis=1)

    # Pick best
    best_idx = jnp.argmin(total_dist)
    best_phase = candidates[best_idx]

    return signal * jnp.exp(-1j * best_phase), best_phase

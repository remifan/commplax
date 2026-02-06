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
from commplax import jax_util as cu
from commplax.buffer import SyncFIFO as FIFO


class _Resampler(NamedTuple):
    init: Callable
    apply: Callable


class VarRateResampler(eqx.Module):
    """Arbitrary-ratio resampler using polyphase filter bank interpolation.

    Converts between sample rates with arbitrary (non-integer) ratios while
    providing anti-aliasing filtering. Commonly used to convert ADC samples
    to symbol-rate samples in optical communication receivers.

    The resampler uses an accumulator-based NCO (Numerically Controlled Oscillator)
    to track fractional sample positions and selects appropriate polyphase
    sub-filters for interpolation.

    Args:
        ratio: Output/input sample rate ratio (f_out / f_in).
            - ratio < 1: Downsampling (e.g., 0.5 for 2:1 decimation)
            - ratio > 1: Upsampling (e.g., 2.0 for 1:2 interpolation)
            - ratio ≈ 1: Rate matching with fine adjustment
        state: Internal filter state (default: initialized from op)
        acc_phase: External phase/rate adjustment ε for timing recovery loops.
            Modifies the effective ratio: Δ = Q/ratio + ε
        op: Polyphase resampler operator (default: var_rate_poly_resampler)
        fifo: Input sample buffer (default: sized for filter length)
        dtype: Complex dtype for signal processing

    Returns:
        When called with input sample x:
        - Updated resampler state
        - Output array of shape (N,) where N = ceil(ratio)
        - Invalid outputs are NaN (occurs during downsampling when input is skipped)

    Example:
        Typical usage for ADC-to-symbol-rate conversion::

            # ADC at 120 GS/s, symbol rate 60 GBaud, want 2 sps output
            sr = 120e9  # ADC sample rate
            br = 60e9   # Baud rate
            sps = 2     # Target samples per symbol
            ratio = sps * br / sr  # = 1.0 (or slightly less with clock mismatch)

            rsplr = VarRateResampler(ratio=ratio)

            # Process with lax.scan
            _, y = lax.scan(lambda r, x: r(x), rsplr, input_signal)

            # Remove NaN outputs (variable-length result)
            y_valid = y[~jnp.isnan(y[:,0]), :]

    Note:
        - For ratio < 1 (downsampling), some inputs produce NaN output
          (skipped to maintain rate). Period ≈ 1/(1-ratio) samples.
        - For ratio > 1 (upsampling), each input produces multiple outputs.
        - The acc_phase parameter enables closed-loop timing recovery by
          fine-tuning the sampling phase on each call.
    """
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
        """Process one input sample.

        Args:
            x: Single input sample (complex scalar)

        Returns:
            Tuple of (updated_resampler, output_array):
            - updated_resampler: New state for next call
            - output_array: Shape (N,) where N=ceil(ratio), NaN for invalid slots
        """
        fifo, _ = self.fifo(x)
        state, y = self.op.apply(self.state, fifo.state[0], self.acc_phase)
        rsplr = dc.replace(self, fifo=fifo, state=state)
        return rsplr, y


def var_rate_poly_resampler(ratio, Q=128, interp_method='nearest neighbor'):
    """Create a polyphase filter bank resampler for arbitrary rate conversion.

    Implements the Farrow-structure variable-rate resampler using a polyphase
    filter bank. An NCO (accumulator) tracks the fractional sample position
    and selects appropriate sub-filters for interpolation.

    Algorithm (per input sample):
        1. Accumulator starts from previous state
        2. For up to N=ceil(ratio) outputs:
           - If acc < Q: output using sub-filter[acc], then acc += Δ
           - If acc >= Q: output NaN (skip), acc unchanged
        3. Wrap accumulator: acc = acc - Q

    The accumulator step Δ = Q/ratio + ε determines the output rate:
        - Δ > Q (ratio < 1): Downsampling, some inputs skipped
        - Δ < Q (ratio > 1): Upsampling, multiple outputs per input
        - Δ = Q (ratio = 1): 1:1 rate, ε provides fine timing adjustment

    Args:
        ratio: Output/input sample rate ratio (f_out / f_in)
        Q: Number of polyphase sub-filters (interpolation resolution).
            Higher Q = finer timing resolution. Default 128.
        interp_method: Sub-filter selection method
            - 'nearest neighbor': Use closest sub-filter (faster)
            - 'two neighbor': Linear interpolation between adjacent sub-filters

    Returns:
        _Resampler namedtuple with:
        - init(subfir_halflen, h, window, dtype): Initialize filter state
        - apply(state, u, ε): Process input sample u with phase adjustment ε

    References:
        [1] Harris, F. "Multirate Signal Processing for Communication Systems"
            Figure 7.32, pp.179

    Note:
        For downsampling (ratio < 1), NaN outputs occur every ~1/(1-ratio) samples.
        For example, ratio=0.997 produces NaN every ~333 samples to maintain rate.
    """
    assert interp_method.lower() in ['nearest neighbor', 'two neighbor']
    ctype = cu.default_complexing_dtype()
    ftype = cu.default_floating_dtype()

    # δ: nominal accumulator step per input sample
    # N: maximum outputs per input (array size for static shape)
    δ = Q / ratio
    N = int(np.ceil(ratio))
    Nzeros = jnp.zeros(N)

    def init(subfir_halflen=10, h=None, window=('kaiser', 5.0), dtype=None):
        """Initialize polyphase filter bank.

        Args:
            subfir_halflen: Half-length of each sub-filter in samples
            h: Custom filter coefficients (default: auto-designed Kaiser)
            window: Window for filter design (default: Kaiser, beta=5.0)
            dtype: Complex dtype for filter coefficients

        Returns:
            state tuple: (hs, acc) where
            - hs: Polyphase filter bank, shape (Q, filter_len)
            - acc: Accumulator initial value (0)
        """
        if h is None:
            # Design lowpass filter with appropriate cutoff for resampling
            max_rate = max(Q, Q / ratio)
            f_c = 1. / max_rate  # Cutoff frequency (relative to Nyquist)
            half_len = subfir_halflen * int(max_rate)
            h = firwin(2 * half_len + 1, f_c, window=window)
        dtype = ctype if dtype is None else dtype
        # Decompose into Q polyphase sub-filters, flip for convolution
        hs = jnp.fliplr(_poly_decomp(h, Q)).astype(dtype) * Q
        acc = jnp.array(0, dtype=ftype)
        state = hs, acc
        return state

    def apply(state, u, ε):
        """Process one input sample through the resampler.

        Args:
            state: (hs, acc) - filter bank and accumulator
            u: Input sample buffer (1D array, length = filter_len)
            ε: Phase/rate adjustment from timing recovery loop.
               Modifies step: Δ = δ + ε

        Returns:
            (new_state, outputs): Updated state and output array of shape (N,).
            Invalid outputs are NaN.
        """
        hs, acc = state
        assert u.ndim == 1 and u.shape[0] == hs.shape[1]

        # Effective step: nominal + external adjustment
        Δ = δ + ε

        # Generate up to N outputs per input
        # acc tracks position within the Q sub-filters
        # When acc < Q, we have a valid output at sub-filter index acc
        # When acc >= Q, no output (NaN), wait for next input
        acc, ind = lax.scan(
            lambda a, _: lax.cond(a < Q, lambda *_: (a+Δ, a), lambda *_: (a, -1.)),
            acc, Nzeros, unroll=True)

        # Wrap accumulator (may wrap multiple times for upsampling)
        acc = acc - Q

        # Pad filter bank with NaN row for invalid indices (-1)
        _hs = jnp.pad(hs, [[0,1],[0,0]], constant_values=jnp.nan)

        # Apply selected sub-filters to input buffer
        match interp_method.lower():
            case 'nearest neighbor':
                # Snap to nearest sub-filter
                ind = jnp.clip(jnp.round(ind).astype(int), max=Q-1)
                v = jnp.dot(_hs[ind], u)
            case 'two neighbor':
                # Linear interpolation between adjacent sub-filters
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
    """Decompose filter into Q polyphase sub-filters.

    Takes a prototype lowpass filter h and splits it into Q sub-filters
    for polyphase implementation. Each sub-filter handles one phase
    of the interpolation.

    Args:
        h: Prototype filter coefficients (1D array)
        Q: Number of polyphase branches (sub-filters)

    Returns:
        hs: Polyphase filter bank, shape (Q, ceil(len(h)/Q))
            Row i contains coefficients for phase i/Q

    Example:
        h = [h0, h1, h2, h3, h4, h5] with Q=3 gives:
        hs = [[h0, h3],
              [h1, h4],
              [h2, h5]]
    """
    h = jnp.atleast_1d(h)
    if not h.ndim == 1:
        raise ValueError("polyphase filter ndim must be 1")
    L = h.shape[0]
    # Pad to multiple of Q, then reshape column-major to get polyphase structure
    hs = jnp.pad(h, [0, Q - L % Q]).reshape((Q, -1), order='F')
    return hs

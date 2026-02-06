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
    """Container for symbol timing synchronization init/apply functions."""
    init: InitFn = None
    apply: ApplyFn = None

    def __iter__(self):
        return iter((self.init, self.apply))


class SymSync(eqx.Module):
    """Symbol timing synchronizer using Gardner TED with interpolation control.

    Recovers symbol timing from a 2 samples-per-symbol (sps) input signal.
    Uses the Gardner Timing Error Detector (TED) and cubic Farrow interpolation
    to track and correct timing offset.

    The synchronizer operates as follows:
    1. Input samples are buffered in a 4-sample FIFO
    2. Cubic interpolation computes samples at fractional delay μ
    3. NCO (η) tracks timing phase, fires strobe at symbol times
    4. Gardner TED computes timing error at symbol strobes
    5. PI loop filter adjusts NCO rate to track timing

    Args:
        kernel: Timing sync kernel (default: symbol_timing_sync())
        state: Initial state tuple (μ, η, strobe, vi, B)
        fifo: Initial sample buffer (default: zeros)
        dtype: Complex dtype for signal processing
        kernel_kwds: Additional arguments for kernel creation

    Returns:
        When called with input sample:
        - Updated SymSync module
        - Tuple (y, e, μ):
            - y: Symbol output (NaN if not at symbol time)
            - e: Timing error from Gardner TED
            - μ: Fractional interpolation delay [0, 1)

    Example:
        Process a 2 sps signal::

            ss = SymSync()
            outputs = []
            for x in signal_2sps:
                ss, (y, e, mu) = ss(x)
                if not jnp.isnan(y):
                    outputs.append(y)

        Or with lax.scan::

            _, (ys, es, mus) = lax.scan(
                lambda s, x: s(x), ss, signal_2sps)
            symbols = ys[~jnp.isnan(ys)]

    Note:
        - Input must be at 2 samples per symbol
        - Output rate is 1 symbol per 2 input samples (on average)
        - Loop converges within ~50-100 symbols typically
    """
    fifo: Array
    state: PyTree
    kernel: PyTree = field(static=True)

    def __init__(self, kernel=None, state=None, fifo=None, dtype=None, kernel_kwds={}):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.kernel = symbol_timing_sync(**kernel_kwds) if kernel is None else kernel
        self.state = self.kernel.init() if state is None else state
        self.fifo = jnp.zeros(4, dtype=dtype) if fifo is None else fifo

    def __call__(self, input):
        """Process one input sample at 2 sps.

        Args:
            input: Single complex sample at 2 samples per symbol

        Returns:
            Tuple of (updated_module, outputs):
            - updated_module: SymSync with updated state
            - outputs: (y, e, μ) where y is symbol (or NaN), e is timing error
        """
        # Shift FIFO and insert new sample
        fifo = jnp.roll(self.fifo, -1, axis=0).at[-1:].set(input)
        # Apply timing sync kernel
        state, out = self.kernel.apply(self.state, fifo)
        ss = dc.replace(self, fifo=fifo, state=state)
        # out = (y, e, μ): symbol output, timing error, fractional delay
        return ss, out


def symbol_timing_sync():
    """Create Gardner-based symbol timing synchronizer for 2 sps signals.

    Implements the Gardner Timing Error Detector (TED) with cubic Farrow
    interpolation for symbol timing recovery. Operates on 2 samples per
    symbol input and outputs 1 symbol per 2 inputs on average.

    Algorithm overview:
        1. **Interpolation**: Cubic Farrow structure computes sample at
           fractional delay μ: xI = (b @ fifo) @ [μ², μ, 1]

        2. **NCO (Numerically Controlled Oscillator)**:
           - η tracks timing phase, decrements by W each sample
           - W = 0.5 + v (nominal rate + loop filter output)
           - When η < 0: symbol strobe fires, η wraps by +1

        3. **Gardner TED**: At symbol time (strobe=2), computes error:
           e = Re{x_mid * (x_prev - x_curr)}
           where x_mid is transition sample, x_prev/x_curr are symbols

        4. **Loop filter**: 2nd-order PI controller
           - vp = K1 * e (proportional)
           - vi += K2 * e (integral)
           - v = vp + vi adjusts NCO rate W

    State variables:
        - μ (mu): Fractional interpolation delay [0, 1)
        - η (eta): NCO timing phase accumulator
        - strobe: Output indicator
            - 0: Idle (between samples)
            - 1: Transition time (midpoint)
            - 2: Symbol time, compute TED (first symbol after transition)
            - 3: Symbol time (subsequent)
        - vi: Loop filter integrator
        - B: TED buffer [transition_sample, prev_symbol_sample]

    Loop gains:
        K1 = -2.46e-3 (proportional)
        K2 = -8.2e-6 (integral)
        These provide ~1% loop bandwidth with critical damping.

    Returns:
        SymbolTimingSync with init() and apply() methods:
        - init(dtype): Initialize state
        - apply(state, fifo): Process 4-sample FIFO, return (new_state, outputs)

    References:
        [1] Rice, Michael. "Digital Communications: A Discrete-Time Approach"
            pp. 493, Gardner timing recovery implementation

    Note:
        - Input must be 2 sps after matched filtering
        - Output y is NaN when strobe < 2 (not at symbol time)
        - Convergence typically within 50-100 symbols
    """
    # PI loop filter gains (negative for correct feedback sign)
    # K1: proportional gain, K2: integral gain
    # These values give ~1% normalized loop bandwidth
    K1 = -2.46e-3
    K2 = -8.2e-6

    # Cubic Farrow interpolation coefficients
    # Computes xI = (b @ x) @ [μ², μ, 1] for fractional delay μ
    # Rows correspond to μ² coefficient, μ coefficient, constant term
    b = 1/2 * jnp.flip(jnp.array(
        [[ 1, -1, -1,  1],   # μ² coefficients
         [-1,  3, -1, -1],   # μ coefficients
         [ 0,  0,  2,  0]],  # constant (μ⁰) coefficients
        dtype=default_complexing_dtype()),
        axis=1)

    def init(dtype=None):
        """Initialize timing synchronizer state.

        Args:
            dtype: Complex dtype for buffers

        Returns:
            State tuple: (μ_next, η_next, strobe, vi, B)
        """
        dtype = default_complexing_dtype() if dtype is None else dtype
        η_next = 0.     # NCO timing phase
        μ_next = 0.     # Fractional interpolation delay
        strobe = 0      # Output strobe indicator
        B = jnp.zeros(2, dtype=dtype)  # TED buffer: [transition, prev_symbol]
        vi = 0.         # Loop filter integrator
        state = μ_next, η_next, strobe, vi, B
        return state

    def apply(state, x):
        """Process one sample through timing synchronizer.

        Args:
            state: (μ_next, η_next, strobe, vi, B) state tuple
            x: 4-sample FIFO buffer (newest sample last)

        Returns:
            (new_state, outputs) where outputs = (y, e, μ):
            - y: Interpolated symbol (NaN if not at symbol time)
            - e: Timing error from Gardner TED
            - μ: Current fractional delay
        """
        μ_next, η_next, strobe, vi, B = state

        μ = μ_next
        η = η_next

        # Cubic interpolation at fractional delay μ
        # m = [μ², μ, 1] for polynomial evaluation
        m = jnp.power(μ, jnp.arange(2,-1,-1))
        xI = jnp.dot(jnp.dot(b, x), m)

        # Output symbol only at strobe (strobe >= 2)
        y = jnp.where(strobe//2 == 1, xI, jnp.nan)

        # Gardner TED: e = Re{mid * (prev - curr)} at symbol strobe
        # B[0] = transition sample, B[1] = previous symbol, xI = current symbol
        e  = jnp.where(
            strobe == 2,
            B[0].real * (B[1].real - xI.real) + B[0].imag * (B[1].imag - xI.imag),
            0.,
            )

        # PI loop filter
        vp = K1 * e           # Proportional term
        vi = vi + K2 * e      # Integral term (accumulates)
        v = vp + vi           # Total loop filter output
        W = 1 / 2 + v         # NCO step: nominal (0.5) + correction

        # Update TED buffer based on strobe state
        # strobe 1,2: shift in new sample (transition or symbol)
        # strobe 0: hold
        # strobe 3: reset prev_symbol
        B = lax.cond(
            (strobe == 1) | (strobe == 2),
            lambda *_: jnp.array([xI, B[0]], dtype=B.dtype),
            lambda *_: lax.cond(
                strobe == 0,
                lambda *_: B,
                lambda *_: jnp.array([xI, 0.], dtype=B.dtype),
            )
        )

        # NCO update: decrement η by step W
        η_next = η - W

        # Strobe logic: when η crosses zero, fire strobe and wrap
        # μ_next = η/W gives fractional delay for interpolation
        η_next, strobe, μ_next = lax.cond(
            η_next < 0,
            lambda *_: (η_next+1, 2+strobe//2, η/W),  # Strobe fired, wrap η
            lambda *_: (η_next,   0+strobe//2, μ),    # No strobe, hold μ
        )

        state = μ_next, η_next, strobe, vi, B
        return state, (y, e, μ)

    return SymbolTimingSync(init, apply)

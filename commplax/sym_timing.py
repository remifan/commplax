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


def centroid_ted(num_taps):
    """Create a centroid-based Timing Error Detector (TED).

    Returns a pure function that computes timing error from an equalizer's
    tap energy centroid. The centroid shifts proportionally to sampling
    phase error, providing a feedback signal for timing recovery.

    The TED accounts for MIMOCell's reversed tap indexing: physical delay
    k corresponds to state index (num_taps - 1 - k).

    Args:
        num_taps: Number of equalizer taps (must match MIMOCell's num_taps)

    Returns:
        A function (eq: MIMOCell) -> timing_error (scalar float).
        Positive error means sampling early (centroid > center).

    Example::

        ted = centroid_ted(11)
        error = ted(mimo_cell)  # scalar timing error
    """
    center = (num_taps - 1) / 2.0
    k = num_taps - 1 - jnp.arange(num_taps)

    def ted(eq):
        # state[0] shape: (dims, dims, num_taps) for up=1,
        # or (dims, dims, num_taps, up) for up>1.
        # Sum energy over all axes except the tap axis (always second-to-last
        # for up>1, or last for up=1). Use dynamic approach: sum all axes
        # except the last, which is always the tap dimension after h_phase
        # reindexing in MIMOCell. For up=1 the shape is (dims, dims, num_taps).
        taps = eq.state[0]
        tap_energy = jnp.sum(jnp.abs(taps) ** 2, axis=tuple(range(taps.ndim - 1)))
        centroid = jnp.sum(k * tap_energy) / (jnp.sum(tap_energy) + 1e-10)
        return centroid - center

    return ted


class TimingLoop(eqx.Module):
    """Closed-loop timing recovery using resampler + MIMO equalizer feedback.

    Composes a VarRateResampler with a MIMOCell equalizer in a feedback loop.
    The equalizer's tap centroid (via a pluggable TED) drives a PI loop
    filter that adjusts the resampler's sampling phase.

    Architecture::

        N sps --> Resampler(eps) --> 1 sps --> MIMOCell --> output
                       ^                          |
                       +--- kp*e + integrator ---+

    The resampler ratio should be set to ``1 / sps`` so that N sps input is
    decimated to 1 sps before the equalizer.  The loop operates
    sample-by-sample: each input sample produces either a valid 1 sps output
    (after resampling + equalization) or NaN (when the resampler skips).
    Feedback uses the *previous* timing error since the resampler must run
    before the equalizer.

    The loop filter is PI (proportional-integral):
    ``eps = -(kp * timing_error + integrator)``, where the integrator
    accumulates ``ki * timing_error`` on each valid output.  With the
    default ``ki=0`` the loop is P-only, which suffices when the nominal
    resampler ratio exactly matches the true sps.  Set ``ki`` to a small
    positive value (e.g. 1e-5) to track residual rate mismatch.

    Args:
        resampler: VarRateResampler with ratio = 1/sps (e.g. 0.5 for 2 sps,
            0.8 for 1.25 sps). The resampler operates on scalar samples;
            for multi-dimensional inputs (dims > 1), resample each dimension
            independently before feeding into this loop.
        equalizer: MIMOCell configured for 1/1 operation (up=1, down=1)
        kp: Proportional loop gain. Default 0.5.
        ki: Integral loop gain. Default 0 (P-only). Use a very small value
            (e.g. 1e-5) when the nominal ratio has residual rate error.
        ted: Timing error detector function (eq -> error). Default centroid_ted.
        timing_error: Initial timing error state. Default 0.
        integrator: Initial integrator state. Default 0.

    Returns:
        When called with a single input sample:
        (updated_loop, (y, timing_error, valid)) where:
        - y: equalized symbol output, shape (dims,) (NaN if resampler skipped)
        - timing_error: current timing error from TED
        - valid: boolean, True if output is valid

    Example::

        from commplax.resampler import VarRateResampler
        from commplax.equalizer import MIMOCell
        from commplax import adaptive_kernel as ak, module as mod

        # 2 sps input
        loop = TimingLoop(
            resampler=VarRateResampler(ratio=0.5),
            equalizer=MIMOCell(11, dims=1, up=1, down=1,
                               kernel=ak.rls_cma(const=const), update_mode=1),
            kp=0.5,
        )
        loop_final, (y, te, valid) = mod.scan_with()(loop, signal_2sps)
        y_valid = y[valid]

        # 1.25 sps with integral term for rate tracking
        loop = TimingLoop(
            resampler=VarRateResampler(ratio=0.8),
            equalizer=MIMOCell(11, dims=1, up=1, down=1,
                               kernel=ak.rls_cma(const=const), update_mode=1),
            kp=0.5,
            ki=1e-5,
        )
    """
    resampler: eqx.Module
    equalizer: eqx.Module
    timing_error: Array
    integrator: Array
    kp: float = field(static=True)
    ki: float = field(static=True)
    ted: Callable = field(static=True)

    def __init__(self, resampler, equalizer, kp=0.5, ki=0.0, ted=None,
                 timing_error=None, integrator=None):
        self.resampler = resampler
        self.equalizer = equalizer
        self.kp = kp
        self.ki = ki
        self.ted = centroid_ted(equalizer.state[0].shape[2]) if ted is None else ted
        self.timing_error = jnp.array(0.0) if timing_error is None else jnp.asarray(timing_error)
        self.integrator = jnp.array(0.0) if integrator is None else jnp.asarray(integrator)

    def __call__(self, x):
        """Process one input sample through the timing loop.

        Args:
            x: Single complex input sample (at N sps)

        Returns:
            Tuple of (updated_loop, (y, timing_error, valid)):
            - y: Equalized output, shape (dims,) (NaN if resampler skipped)
            - timing_error: TED output after equalization
            - valid: True if a symbol was produced
        """
        dims = self.equalizer.fifo.shape[1]

        # 1. PI loop filter: eps from previous timing error + integrator
        #    positive error -> sampling early -> negative eps -> delay output
        eps = -(self.kp * self.timing_error + self.integrator)

        # 2. Resample with phase adjustment
        rsplr = dc.replace(self.resampler, acc_phase=eps)
        rsplr_new, y_arr = rsplr(x)
        y_1sps = y_arr[0]
        valid = ~jnp.isnan(y_1sps)

        # 3. Conditionally run equalizer + TED (skip on NaN to protect state)
        def _update(eq, te):
            eq_new, y_eq = eq((jnp.atleast_1d(y_1sps), jnp.zeros(dims)))
            te_new = self.ted(eq_new)
            return eq_new, y_eq, te_new

        def _hold(eq, te):
            return eq, jnp.full(dims, jnp.nan + 0j, dtype=eq.fifo.dtype), te

        eq_new, y_eq, te_new = lax.cond(valid, _update, _hold, self.equalizer, self.timing_error)

        # 4. Update integrator (only on valid outputs)
        integrator_new = jnp.where(valid,
                                   self.integrator + self.ki * te_new,
                                   self.integrator)

        # 5. Return updated loop and outputs
        loop = dc.replace(self, resampler=rsplr_new, equalizer=eq_new,
                          timing_error=te_new, integrator=integrator_new)
        return loop, (y_eq, te_new, valid)

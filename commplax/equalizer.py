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

"""Adaptive equalization and carrier recovery modules for coherent receivers.

This module provides equinox-based modules for the DSP chain in coherent
optical and wireless communication receivers:

- **MIMOCell**: Multi-input multi-output adaptive equalizer with polyphase
  rational-rate resampling support. Handles chromatic dispersion compensation,
  polarization demultiplexing, and timing recovery in coherent optical systems.

- **FOE**: Frequency Offset Estimator for carrier frequency recovery.
  Compensates laser frequency mismatch between transmitter and receiver.

- **CPR**: Carrier Phase Recovery using phase-locked loop (PLL) structure.
  Tracks residual phase noise after frequency offset correction.

- **align_phase**: Phase ambiguity resolver for post-CPR alignment.

Polyphase Resampling:
    MIMOCell supports arbitrary rational up/down resampling ratios through
    polyphase decomposition. Helper functions:
    - rat_poly_inp_phases: Computes input sample phases for rate conversion
    - rat_poly_h_phases: Computes filter coefficient indices for polyphase

Typical DSP Chain:
    1. VarRateResampler (ADC rate → 2 sps)
    2. MIMOCell with CMA/RLS (chromatic dispersion, pol demux)
    3. FOE (frequency offset estimation/compensation)
    4. CPR (carrier phase recovery)
    5. align_phase (resolve π/2 ambiguity)
    6. Symbol decision and decoding

Example:
    Dual-polarization coherent receiver::

        from commplax import adaptive_kernel as ak
        from commplax.equalizer import MIMOCell, FOE, CPR, align_phase

        # 2x2 MIMO equalizer with CMA-RLS
        eq = MIMOCell(num_taps=21, dims=2, kernel=ak.rls_cma())

        # Frequency offset estimator
        foe = FOE(mode="feedforward")

        # Carrier phase recovery (per polarization)
        cpr = CPR(kernel=ak.cpr_4thpower_pll(mu=0.01))

        # Process through chain
        eq, y_eq = lax.scan(lambda e, x: e(x), eq, signal_2pol)
        foe, y_foe = foe(y_eq)
        cpr, y_cpr = lax.scan(lambda c, y: c(y), cpr, y_foe)
        y_aligned, phase = align_phase(y_cpr, constellation)
"""

import dataclasses as dc
import numpy as np
from jax import lax, numpy as jnp, debug as jdbg
import equinox as eqx
from equinox import field
from commplax import adaptive_kernel as _ak
from commplax.jax_util import default_complexing_dtype, default_floating_dtype, astuple
from commplax._deprecated.cxopt import make_schedule, Union, Schedule
from functools import lru_cache
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any, TypeVar
from numpy.typing import ArrayLike


#@lru_cache  #lru_cache may conflict with jax.jit(jax.pmap)
def rat_poly_inp_phases(up: int, down: int):
    """Compute input sample phases for polyphase rational resampling.

    For a rational resampling ratio up/down, determines which input samples
    contribute to outputs. This implements the commutator timing for a
    polyphase filterbank operating at rate up/down.

    Algorithm:
        Starting from filter phase t=0 and input index x_i=0, advance through
        the polyphase structure. For each output, record which input sample
        is needed, then advance: t += down, x_i += t // up, t = t % up.
        Continue until we've covered 'down' input samples (one output period).

    Args:
        up: Upsampling factor (interpolation rate)
        down: Downsampling factor (decimation rate)

    Returns:
        List of input indices that produce valid outputs within one period.
        Length = up (number of outputs per 'down' inputs).

    Examples:
        >>> rat_poly_inp_phases(1, 1)  # 1:1, no resampling
        [0]  # Each input produces one output

        >>> rat_poly_inp_phases(2, 1)  # 2:1 upsampling
        [0, 0]  # One input produces two outputs (both use same input)

        >>> rat_poly_inp_phases(1, 2)  # 1:2 downsampling
        [0]  # Every other input produces output

        >>> rat_poly_inp_phases(3, 2)  # 3:2 interpolation
        [0, 0, 1]  # Three outputs from two inputs

    References:
        scipy.signal._upfirdn_apply.pyx (polyphase resampling implementation)
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_upfirdn_apply.pyx#L458

    Note:
        Result is used by MIMOCell to determine when to consume input samples.
        The outer_i counter mod down is compared against these phases.
    """
    _gcd = int(np.gcd(up, down))
    up //= _gcd
    down //= _gcd
    t = 0  # filter phase (which polyphase subfilter)
    x_i = 0  # input index
    x_I = []
    while x_i < down:
        x_I.append(x_i)
        t += down
        x_i += t // up  # advance input when phase wraps
        t = t % up
    return x_I


#@lru_cache  #lru_cache may conflict with jax.jit(jax.pmap)
def rat_poly_h_phases(h_len: int, up: int, flipped: bool = True):
    """Compute filter coefficient indices for polyphase decomposition.

    Decomposes a filter of length h_len into 'up' polyphase subfilters.
    Returns indices arranged for efficient polyphase convolution, with
    optional time-reversal for FIR filtering (convolution vs correlation).

    The polyphase decomposition splits filter h[n] into 'up' subfilters:
        h_k[m] = h[m * up + k]  for k = 0, 1, ..., up-1

    For upsampling, each output uses a different subfilter based on its
    phase within the upsampling period.

    Args:
        h_len: Length of the prototype filter
        up: Number of polyphase branches (upsampling factor)
        flipped: If True (default), reverse subfilter order for convolution.
            Use flipped=True for FIR filtering, False for correlation.

    Returns:
        Array of shape (up, ceil(h_len/up)) containing indices into the
        original filter. Index = h_len indicates padding (out of bounds).

    Example:
        For h_len=7, up=2:
            Original: h = [h0, h1, h2, h3, h4, h5, h6]
            Polyphase (column-major fill, then flip):
                Phase 0: [h6, h4, h2, h0]  (indices [6, 4, 2, 0])
                Phase 1: [h7, h5, h3, h1]  (indices [7, 5, 3, 1], 7=pad)

        >>> h_phase = rat_poly_h_phases(7, 2)
        >>> h_phase[0]  # Phase 0 indices (flipped)
        array([6, 4, 2, 0])
        >>> h_phase[1]  # Phase 1 indices (7 is padding/invalid)
        array([7, 5, 3, 1])

    References:
        scipy.signal._upfirdn.py (polyphase filter structure)
        https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/_upfirdn.py#L46

    Note:
        Used by MIMOCell to select which filter coefficients to use based
        on the current output phase (inner_i % up). The flipped arrangement
        allows direct dot product with the FIFO buffer.
    """
    h_padlen = -h_len % up  # padding to make length divisible by up
    # Create index array, pad with h_len (out-of-bounds marker)
    ind = jnp.pad(jnp.arange(h_len), (0, h_padlen), constant_values=h_len)
    # Reshape column-major into (up, subfilter_len), optionally flip
    ind = ind.reshape(
        (up, ind.shape[-1] // up), order='F'
    )[..., ::1 - 2 * int(flipped)]  # flip if flipped=True
    return ind


def ak_update_flag(umode, train, i):
    """Determine whether to update adaptive filter coefficients.

    Implements the update logic for adaptive kernels based on:
    - Update mode schedule (frozen, decision-directed, training-only)
    - Training signal availability

    Update modes:
        - umode(i) = 0: Frozen - no updates, use current coefficients
        - umode(i) = 1: Decision-directed - update using hard decisions
        - umode(i) = 2: Training-only - update only when training data available

    Args:
        umode: Schedule function returning update mode at iteration i
        train: Schedule function returning True if training data available at i
        i: Current iteration index

    Returns:
        Boolean indicating whether to update filter coefficients.
        True if: (not frozen) AND (training OR decision-directed)
    """
    cond0 = umode(i) > 0  # not frozen
    cond1 = train(i)  # training data available
    cond2 = ~train(i) & (umode(i) == 1)  # decision-directed mode
    return cond0 & (cond1 | cond2)


class MIMOCell(eqx.Module):
    """MIMO adaptive equalizer with polyphase rational-rate support.

    Implements a multi-input multi-output (MIMO) adaptive FIR equalizer
    supporting arbitrary rational up/down resampling ratios. Uses polyphase
    decomposition for efficient fractional-rate equalization.

    The equalizer processes samples through an adaptive FIR filter with
    dimensions (dims × dims), supporting 2×2 for dual-polarization coherent
    receivers or N×N for spatial multiplexing systems.

    Polyphase Operation:
        For up/down ratio resampling, the filter is decomposed into 'up'
        polyphase subfilters. The NCO-like counters (inner_i, outer_i) track:
        - outer_i: Input sample counter (mod down determines input consumption)
        - inner_i: Output sample counter (mod up determines filter phase)

        For up=2, down=1 (2× interpolation):
        - Each input produces 2 outputs using alternating filter phases
        - inner_i advances twice per input, cycling through h_phase[0], h_phase[1]

        For up=1, down=2 (2× decimation):
        - Every other input is skipped (in_phase = [0])
        - Only outer_i % 2 == 0 produces output

    Filter State Layout:
        state[0] has shape (dims, dims, subfilter_len, up)
        - dims × dims: MIMO butterfly structure
        - subfilter_len: Number of taps per polyphase branch
        - up: Number of polyphase phases

        To access full filter: state[0][..., h_phase[phase]] selects
        coefficients for the given phase, reordering for convolution.

    Attributes:
        fifo: Input sample buffer, shape (subfilter_len, dims)
        state: Adaptive kernel state tuple (weights, aux_states...)
        inner_i: Output sample counter (determines filter phase)
        outer_i: Input sample counter (determines input consumption)
        in_phase: Array of valid input phases from rat_poly_inp_phases
        h_phase: Filter coefficient indices from rat_poly_h_phases
        up: Upsampling factor (interpolation)
        down: Downsampling factor (decimation)
        kernel: Adaptive algorithm (LMS, RLS, CMA, etc.)
        update_mode: Update schedule (0=frozen, 1=decision, 2=training-only)

    Example:
        Basic 2×2 MIMO equalizer at 1 sps::

            from commplax import adaptive_kernel as ak

            # CMA-RLS equalizer with 21 taps, dual-pol
            eq = MIMOCell(
                num_taps=21,
                dims=2,
                kernel=ak.rls_cma(lam=0.999, delta=0.01),
            )

            # Process with lax.scan
            eq, y = lax.scan(lambda e, x: e(x), eq, signal_2pol)

        With fractional-rate (2 sps to 1 sps)::

            eq = MIMOCell(num_taps=21, dims=2, up=1, down=2)
            # Input at 2 sps, output at 1 sps (every other sample)

    Note:
        - For up=down=1 (standard case), one output per input
        - Tap centroid of state[0] can estimate timing offset
        - Filter is stored in reversed order; tap k corresponds to delay
          (num_taps - 1 - k) samples
    """
    fifo: Array
    state: Array
    inner_i: Array
    outer_i: Array
    in_phase: Array
    h_phase: Array
    up: int = field(static=True)
    down: int = field(static=True)
    kernel: PyTree = field(static=True)
    update_mode: Union[int, Schedule] = field(static=True)

    def __init__(
        self,
        num_taps: int = 15,
        dims: int = 1,
        dtype=None,
        w0=None,
        up: int = 1,
        down: int = 1,
        kernel: PyTree = None,
        state: Array = None,
        fifo: Array = None,
        inner_i: Array = jnp.array(0),
        outer_i: Array = jnp.array(0),
        in_phase: Array = None,
        h_phase: Array = None,
        update_mode: Union[int, Schedule] = 1,
    ):
        """Initialize MIMO equalizer.

        Args:
            num_taps: Number of filter taps (default: 15)
            dims: MIMO dimensions, e.g., 2 for dual-pol (default: 1)
            dtype: Complex dtype for filter coefficients
            w0: Initial filter weights (default: identity-like initialization)
            up: Upsampling factor for output rate (default: 1)
            down: Downsampling factor for input rate (default: 1)
            kernel: Adaptive algorithm from adaptive_kernel (default: LMS)
            state: Pre-initialized kernel state (default: from kernel.init)
            fifo: Pre-initialized input buffer (default: zeros)
            inner_i: Initial output counter (default: 0)
            outer_i: Initial input counter (default: 0)
            in_phase: Custom input phase schedule (default: computed)
            h_phase: Custom filter phase indices (default: computed)
            update_mode: Filter update schedule (default: 1 = decision-directed)
        """
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.up = up
        self.down = down
        self.kernel = _ak.lms() if kernel is None else kernel
        # Initialize kernel state with polyphase structure (nspike=up for phase count)
        self.state = self.kernel.init(w0=w0, taps=num_taps, dims=dims, dtype=dtype, nspike=up) if state is None else state
        self.inner_i = jnp.asarray(inner_i) if inner_i is None else inner_i
        self.outer_i = jnp.asarray(outer_i) if outer_i is None else outer_i
        # Compute polyphase schedules if not provided
        self.in_phase = jnp.asarray(rat_poly_inp_phases(up, down)) if in_phase is None else in_phase
        self.h_phase = jnp.asarray(rat_poly_h_phases(num_taps, up)) if h_phase is None else h_phase
        # FIFO length = subfilter length (h_phase.shape[-1])
        self.fifo = jnp.zeros((self.h_phase.shape[-1], dims), dtype=dtype) if fifo is None else fifo
        self.update_mode = make_schedule(update_mode)

    def __call__(self, input: PyTree):
        """Process one input sample through the MIMO equalizer.

        Args:
            input: Input sample(s) with optional auxiliary data.
                - For MIMO: shape (dims,) complex array
                - For training: (x, d) tuple where d is desired output
                - For CMA/RLS-CMA: just x (blind equalization)

        Returns:
            Tuple of (updated_cell, output):
            - updated_cell: MIMOCell with updated state
            - output: Equalized output, shape (dims,). Zero if input skipped
              (for down > 1 decimation when not at valid phase).
        """
        x, *args = astuple(input)
        # Shift FIFO and insert new input sample
        fifo = jnp.roll(self.fifo, -1, axis=0).at[-1:].set(x)

        # Select filter coefficients for current output phase
        # h_phase[phase] contains indices into full filter for this phase
        i_w = self.h_phase[self.inner_i % self.up]

        # Check if current input phase produces an output
        # For down=2: only outer_i % 2 == 0 is valid (skip every other)
        valid_input_phase = jnp.any(self.outer_i % self.down == self.in_phase)

        # Extract subfilter coefficients for current phase
        # state[0] has shape (dims, dims, subfilter_len, up)
        # Indexing with i_w reorders to (dims, dims, subfilter_len)
        w_i = self.state[0].at[..., i_w].get(mode='fill', fill_value=0.)
        state_i = (w_i,) + self.state[1:]

        # Apply filter only at valid input phases
        dummy_out = jnp.zeros_like(x)
        output, inner_i = lax.cond(
            valid_input_phase,
            lambda *_: (self.kernel.apply(state_i, fifo), self.inner_i + 1),
            lambda *_: (dummy_out, self.inner_i),
        )

        # Update filter coefficients if conditions met
        # (valid input, update mode active, training or decision-directed)
        state_i = lax.cond(
            valid_input_phase & ak_update_flag(self.update_mode, self.kernel.update.train, inner_i),
            lambda *_: self.kernel.update(self.inner_i, state_i, (fifo, *args))[0],
            lambda *_: state_i,
        )

        # Write updated coefficients back to full state
        w_i = self.state[0].at[..., i_w].set(state_i[0])
        state = (w_i,) + state_i[1:]

        # Create updated cell with new state and incremented counters
        cell = dc.replace(self, fifo=fifo, state=state, inner_i=inner_i, outer_i=self.outer_i + 1)
        return cell, output


#TODO tap profiler to track clock drift


class FOE(eqx.Module):
    """Frequency Offset Estimator for carrier recovery.

    Estimates and compensates carrier frequency offset (CFO) in coherent
    optical or wireless receivers. Supports feedforward and feedback modes.

    Frequency offset appears as a linear phase rotation over time:
        y[n] = x[n] * exp(j * 2π * f_offset * n)

    The FOE estimates f_offset and applies the inverse rotation to
    derotate the signal back to baseband.

    Modes:
        - "feedforward": Estimate offset from block, then apply correction.
            Order: update() → apply(). Good for known/stable offsets.
        - "feedback": Apply current estimate, then update from corrected signal.
            Order: apply() → update(). Better for tracking varying offsets.

    Attributes:
        fo: Current frequency offset estimate (radians per sample)
        i: Update iteration counter
        t: Sample time counter (for phase accumulation)
        state: Kernel state for estimation algorithm
        kernel: FOE algorithm (default: Extended Kalman Filter based)
        uar: Update-to-apply ratio scaling factor
        mode: "feedforward" or "feedback"

    Example:
        Basic frequency offset compensation::

            foe = FOE(fo=0.0, mode="feedforward")

            # Process signal block
            foe, corrected = foe(signal_block)

            # Check estimated offset
            print(f"Estimated FO: {foe.fo / (2*np.pi):.6f} cycles/sample")

        With known initial offset::

            # Initial estimate from acquisition
            foe = FOE(fo=0.001, mode="feedback")

    Note:
        - fo is in radians per sample; convert to Hz: fo * sample_rate / (2π)
        - For optical systems, typical offsets are < 1 GHz at symbol rate
        - uar (update-to-apply ratio) handles rate differences between
          estimation rate and correction rate
    """
    fo: Float
    i: Int
    t: Int
    state: PyTree
    kernel: PyTree = field(static=True)
    uar: float = field(static=True)
    mode: str = field(static=True)

    def __init__(self, fo=0.0, uar=1.0, kernel=None, i=0, t=0, mode="feedforward", state=None, kernel_kwds={}):
        """Initialize frequency offset estimator.

        Args:
            fo: Initial frequency offset estimate (radians/sample, default: 0.0)
            uar: Update-to-apply ratio for rate conversion (default: 1.0)
            kernel: FOE algorithm kernel (default: foe_YanW_ekf)
            i: Initial iteration counter (default: 0)
            t: Initial time counter (default: 0)
            mode: "feedforward" or "feedback" (default: "feedforward")
            state: Pre-initialized kernel state (default: from kernel.init)
            kernel_kwds: Additional arguments for kernel initialization
        """
        self.i = jnp.asarray(i)
        self.t = jnp.asarray(t)
        self.kernel = _ak.foe_YanW_ekf(**kernel_kwds) if kernel is None else kernel
        self.fo = jnp.asarray(fo)
        self.uar = uar * 1.0
        # For feedforward, initialize kernel with known offset; for feedback, start at 0
        fo4init = fo if mode == "feedforward" else 0.
        self.state = self.kernel.init(fo4init) if state is None else state
        self.mode = mode

    def __call__(self, input):
        """Process signal block through FOE.

        Args:
            input: Complex signal array, shape (N,) or (N, dims)

        Returns:
            Tuple of (updated_foe, corrected_signal)
        """
        if self.mode == "feedforward":
            # Feedforward: estimate first, then correct
            foe = self.update(input)[0]
            foe, output = foe.apply(input)
        else:
            # Feedback: correct with current estimate, then update
            foe, output = self.apply(input)
            foe = foe.update(output)[0]
        return foe, output

    def update(self, input):
        """Update frequency offset estimate from signal block.

        Uses the kernel's estimation algorithm (e.g., EKF) to refine
        the frequency offset estimate.

        Args:
            input: Complex signal array for estimation

        Returns:
            Tuple of (updated_foe, input) - input passed through unchanged
        """
        state, out = self.kernel.update(self.i, self.state, input)
        # For feedback mode, accumulate offset; for feedforward, replace
        fo = self.fo + out[0] if self.mode == "feedback" else out[0]
        foe = dc.replace(self, fo=fo, state=state, i=self.i + 1)
        return foe, input

    def apply(self, input):
        """Apply frequency correction to signal block.

        Derotates the signal by the estimated frequency offset:
            output[n] = input[n] * exp(-j * fo * n)

        Args:
            input: Complex signal array, shape (N,) or (N, dims)

        Returns:
            Tuple of (updated_foe, corrected_signal)
        """
        # Generate time indices for this block
        T = self.t + jnp.arange(input.shape[0])
        # Scale offset by update-to-apply ratio
        fo = self.fo * self.uar
        # Apply frequency derotation
        output = input * jnp.exp(-1j * fo * T)
        # Update time counter for next block
        foe = dc.replace(self, t=T[-1] + 1)
        return foe, output


class CPR(eqx.Module):
    """Single-dimensional Carrier Phase Recovery module (symbol-by-symbol).

    Tracks and compensates residual carrier phase noise after frequency
    offset estimation. Uses a phase-locked loop (PLL) structure with
    the 4th-power algorithm by default for QAM signals.

    The 4th-power algorithm raises the signal to the 4th power to remove
    data modulation (for 4-point symmetry constellations like QPSK, 16QAM),
    then extracts the phase error from arg(y^4)/4.

    Phase Ambiguity:
        4th-power CPR has inherent π/2 phase ambiguity (±90°, ±180°).
        Use align_phase() after CPR to resolve this ambiguity using
        the known constellation.

    Attributes:
        phase: Current phase estimate (radians)
        i: Symbol counter
        kernel: CPR algorithm (default: cpr_4thpower_pll)

    Example:
        Single-polarization CPR::

            cpr = CPR(kernel=ak.cpr_4thpower_pll(mu=0.01))
            cpr, corrected = lax.scan(lambda c, y: c(y), cpr, symbols)

        Dual-polarization using vmap::

            @eqx.filter_vmap
            def make_cpr_ensemble(_):
                return CPR(kernel=ak.cpr_4thpower_pll(mu=0.01))
            cpr = make_cpr_ensemble(jnp.arange(2))  # 2 independent CPRs

            @eqx.filter_vmap
            def cpr_step(cpr, x):
                return cpr(x)

            # Process dual-pol signal: shape (N, 2)
            cpr, out = mod.scan_with(cpr_step)(cpr, signal)

        Resolve phase ambiguity after CPR::

            aligned, phase = align_phase(out, sym_map.const('16QAM', norm=True))

    Note:
        - Process at 1 sample per symbol (after matched filter and timing)
        - mu (loop gain) controls tracking bandwidth vs noise rejection
        - Higher mu = faster tracking, more noise; lower mu = smoother, slower
        - Typical mu: 0.01-0.05 for optical coherent receivers
    """
    phase: Float
    i: Int
    kernel: PyTree = field(static=True)

    def __init__(self, phase=0.0, kernel=None, i=0):
        """Initialize carrier phase recovery module.

        Args:
            phase: Initial phase estimate in radians (default: 0.0)
            kernel: CPR algorithm kernel (default: cpr_4thpower_pll)
            i: Initial symbol counter (default: 0)
        """
        self.i = jnp.asarray(i)
        self.phase = jnp.asarray(phase)
        self.kernel = _ak.cpr_4thpower_pll() if kernel is None else kernel

    def __call__(self, y):
        """Process single symbol: update phase and apply correction.

        Args:
            y: Single complex symbol

        Returns:
            Tuple of (updated_cpr, corrected_symbol)
        """
        new_phase, _ = self.kernel.update(self.i, self.phase, y)
        y_corrected = self.kernel.apply(new_phase, y)
        cpr = dc.replace(self, phase=new_phase, i=self.i + 1)
        return cpr, y_corrected

    def update(self, y):
        """Update phase estimate from single symbol (no correction applied).

        Args:
            y: Single complex symbol

        Returns:
            Tuple of (updated_cpr, original_symbol)
        """
        new_phase, aux = self.kernel.update(self.i, self.phase, y)
        cpr = dc.replace(self, phase=new_phase, i=self.i + 1)
        return cpr, y

    def apply(self, y):
        """Apply current phase correction to single symbol (no update).

        Args:
            y: Single complex symbol

        Returns:
            Tuple of (unchanged_cpr, corrected_symbol)
        """
        y_corrected = self.kernel.apply(self.phase, y)
        return self, y_corrected


def align_phase(signal, const, n_samples=1000):
    """Resolve phase ambiguity by testing candidate rotations.

    After carrier phase recovery (CPR), there is often a π/2 phase ambiguity
    due to the 4th-power algorithm. This function tests 4 candidate phases
    (0°, 90°, 180°, 270°) and selects the one that minimizes total distance
    to the reference constellation points.

    Algorithm:
        1. Take first n_samples for efficiency
        2. Rotate signal by each candidate phase
        3. Compute distance to nearest constellation point
        4. Select phase with minimum total squared distance

    Args:
        signal: Complex signal array, shape (..., N) or (N,)
        const: Reference constellation points as 1D array
            (e.g., sym_map.const('16QAM', norm=True))
        n_samples: Number of samples to use for alignment (default: 1000)
            Using fewer samples speeds up computation.

    Returns:
        Tuple of (aligned_signal, best_phase):
        - aligned_signal: Input signal rotated by best_phase
        - best_phase: The phase offset applied (0, π/2, π, or 3π/2)

    Example:
        After CPR with π/2 ambiguity::

            from commplax import sym_map

            # Get reference constellation
            const = sym_map.const('16QAM', norm=True)

            # Align phase after CPR
            aligned, phase = align_phase(cpr_output, const)
            print(f"Applied rotation: {np.degrees(phase):.0f}°")

    Note:
        - Only works for constellations with 90° rotational symmetry (QAM)
        - For 8PSK or other constellations, modify the candidate phases
        - Uses first n_samples for speed; ensure signal is representative
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

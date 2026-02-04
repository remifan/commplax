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

"""Sub-carrier multiplexing/demultiplexing kernels.

Used in 1600ZR+ for dual digital sub-carrier transmission.
Each sub-carrier carries a DSP super-frame, then they are RRC filtered
and frequency-offset before combining.

Reference:
    [1] OIF 1600ZR+ Implementation Agreement (oif2024.447.06), Section 6.8
"""

import jax.numpy as jnp
from jax import vmap
from commplax.filter import rcosdesign
from commplax.signal import fftconvolve


def _upsample(x, factor):
    """Upsample 1D signal by inserting zeros."""
    n = x.shape[0]
    y = jnp.zeros(n * factor, dtype=x.dtype)
    y = y.at[::factor].set(x)
    return y


def _upsample_2d(x, factor):
    """Upsample 2D signal (n_samples, n_pols) along first axis."""
    return vmap(lambda col: _upsample(col, factor), in_axes=1, out_axes=1)(x)


def _apply_filter(x, h):
    """Apply filter to 1D signal."""
    return fftconvolve(x, h, mode='same')


def _apply_filter_2d(x, h):
    """Apply filter to 2D signal (n_samples, n_pols) along first axis."""
    if x.ndim == 1:
        return _apply_filter(x, h)
    return vmap(lambda col: _apply_filter(col, h), in_axes=1, out_axes=1)(x)


def subcarrier_mux(num_subcarriers=2, rolloff=1/16, sps=2, rrc_span=32):
    """
    Sub-carrier multiplexer/demultiplexer kernel for ZR+.

    Multiplexes digital sub-carriers with RRC pulse shaping and frequency offset:
    - Each sub-carrier is upsampled and RRC filtered (roll-off = 1/16)
    - Sub-carriers are offset from center by ±¼·Fb·(1+α)
    - Signals are summed to form the combined output

    Args:
        num_subcarriers: Number of sub-carriers (default: 2 for ZR+)
        rolloff: RRC roll-off factor α (default: 1/16)
        sps: Samples per symbol after upsampling (default: 2)
        rrc_span: RRC filter span in symbols (default: 32)

    Returns:
        mux: Function (subcarrier_symbols) -> combined_signal
        demux: Function (combined_signal) -> subcarrier_symbols

    Example:
        >>> mux, demux = subcarrier_mux(num_subcarriers=2, sps=2)
        >>> # TX: multiplex two sub-carriers
        >>> combined = mux([sc0_symbols, sc1_symbols])
        >>> # RX: demultiplex back
        >>> sc0_rx, sc1_rx = demux(combined)

    Note:
        Sub-carrier frequency offsets per OIF 1600ZR+ spec (Section 6.8):
        - SC0: -¼·Fb·(1+α) from center
        - SC1: +¼·Fb·(1+α) from center
        where Fb is the aggregate baud rate.
    """
    # Design RRC filter for pulse shaping
    rrc = jnp.array(rcosdesign(rolloff, rrc_span, sps, shape='sqrt'))

    # Normalized frequency offset for each sub-carrier
    # Sub-carriers at ±¼·(1+α) relative to aggregate symbol rate
    # In normalized frequency (relative to sample rate = sps * baud_per_sc):
    # offset = ±¼·(1+α) * Fb / Fs = ±¼·(1+α) * Fb / (sps * Fb/2) = ±(1+α)/(2*sps)
    freq_offset_norm = (1 + rolloff) / (2 * sps)

    def _freq_shift(signal, f_norm):
        """Apply frequency shift to signal."""
        n_samples = signal.shape[0]
        t = jnp.arange(n_samples)
        shift = jnp.exp(1j * 2 * jnp.pi * f_norm * t)
        if signal.ndim == 2:
            shift = shift[:, None]
        return signal * shift

    def mux(subcarrier_symbols):
        """
        Multiplex sub-carrier symbols into combined signal.

        Args:
            subcarrier_symbols: List of symbol arrays, each (n_symbols,) or (n_symbols, n_pols)
                               or array of shape (num_subcarriers, n_symbols, n_pols)

        Returns:
            combined: Combined signal, shape (n_symbols * sps, n_pols) or (n_symbols * sps,)
        """
        if isinstance(subcarrier_symbols, jnp.ndarray) and subcarrier_symbols.ndim == 3:
            subcarrier_symbols = [subcarrier_symbols[i] for i in range(num_subcarriers)]

        combined = None
        for i, sc_syms in enumerate(subcarrier_symbols):
            sc_syms = jnp.atleast_1d(sc_syms)
            is_1d = sc_syms.ndim == 1

            if is_1d:
                sc_syms = sc_syms[:, None]

            # 1. Upsample (insert zeros)
            upsampled = _upsample_2d(sc_syms, sps)

            # 2. RRC pulse shaping
            shaped = _apply_filter_2d(upsampled, rrc)

            # 3. Frequency shift: SC0 -> -offset, SC1 -> +offset
            f_shift = (2 * i - (num_subcarriers - 1)) / (num_subcarriers - 1) * freq_offset_norm
            shifted = _freq_shift(shaped, f_shift)

            # 4. Accumulate
            if combined is None:
                combined = shifted
            else:
                combined = combined + shifted

            if is_1d:
                combined = combined[:, 0]

        return combined

    def demux(combined_signal):
        """
        Demultiplex combined signal into sub-carrier symbols.

        Args:
            combined_signal: Combined signal, shape (n_samples,) or (n_samples, n_pols)

        Returns:
            subcarrier_symbols: List of symbol arrays
        """
        combined = jnp.atleast_1d(combined_signal)
        is_1d = combined.ndim == 1

        if is_1d:
            combined = combined[:, None]

        subcarrier_symbols = []
        for i in range(num_subcarriers):
            # 1. Frequency shift to baseband (opposite sign of mux)
            f_shift = -(2 * i - (num_subcarriers - 1)) / (num_subcarriers - 1) * freq_offset_norm
            baseband = _freq_shift(combined, f_shift)

            # 2. Matched filter (RRC)
            filtered = _apply_filter_2d(baseband, rrc)

            # 3. Downsample (at optimal sampling instant)
            # Adjust for filter group delay
            delay = len(rrc) // 2
            symbols = filtered[delay::sps]

            if is_1d:
                symbols = symbols[:, 0]

            subcarrier_symbols.append(symbols)

        return subcarrier_symbols

    return mux, demux


def subcarrier_distribute(num_subcarriers=2, interleave_bits=128):
    """
    Distribute OFEC output bits to sub-carriers.

    After OFEC interleaving, bits are distributed to sub-carriers in groups
    of 128 bits in round-robin fashion.

    Args:
        num_subcarriers: Number of sub-carriers (default: 2)
        interleave_bits: Bits per distribution group (default: 128)

    Returns:
        distribute: Function (bits) -> tuple of bit arrays
        merge: Function (tuple of bit arrays) -> bits

    Example:
        >>> distribute, merge = subcarrier_distribute()
        >>> sc0_bits, sc1_bits = distribute(ofec_output)
        >>> recovered = merge((sc0_bits, sc1_bits))
    """

    def distribute(bits):
        """
        Distribute bits to sub-carriers in round-robin groups.

        Args:
            bits: Input bits from OFEC interleaver, shape (n_bits,)

        Returns:
            subcarrier_bits: Tuple of bit arrays, one per sub-carrier
        """
        bits = jnp.atleast_1d(bits)
        n_groups = bits.shape[0] // (interleave_bits * num_subcarriers)
        n_total = n_groups * interleave_bits * num_subcarriers

        # Reshape to (n_groups, num_subcarriers, interleave_bits)
        bits_grouped = bits[:n_total].reshape(n_groups, num_subcarriers, interleave_bits)

        # Extract each sub-carrier's bits
        subcarrier_bits = tuple(
            bits_grouped[:, i, :].reshape(-1) for i in range(num_subcarriers)
        )

        return subcarrier_bits

    def merge(subcarrier_bits):
        """
        Merge sub-carrier bits back to single stream.

        Args:
            subcarrier_bits: Tuple of bit arrays from each sub-carrier

        Returns:
            bits: Merged bit stream
        """
        # Stack and interleave
        sc_arrays = [jnp.atleast_1d(sb).reshape(-1, interleave_bits) for sb in subcarrier_bits]
        n_groups = sc_arrays[0].shape[0]

        # Stack to (n_groups, num_subcarriers, interleave_bits) and flatten
        stacked = jnp.stack(sc_arrays, axis=1)
        bits = stacked.reshape(-1)

        return bits

    return distribute, merge

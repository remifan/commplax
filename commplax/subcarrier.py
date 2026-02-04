# Copyright 2025 The Commplax Authors.
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
from jax import lax


def subcarrier_mux(num_subcarriers=2, rolloff=1/16, baud_rate=252.5e9):
    '''
    Sub-carrier multiplexer kernel for ZR+.

    Multiplexes two digital sub-carriers onto dual polarization:
    - Each sub-carrier is RRC filtered (roll-off = 1/16)
    - Sub-carriers are offset from center by ±¼·Fb·(1+α)
    - X and Y polarizations are merged

    Args:
        num_subcarriers: Number of sub-carriers (default: 2 for ZR+)
        rolloff: RRC roll-off factor (default: 1/16)
        baud_rate: Aggregate baud rate in Hz (default: 252.5 GBd)

    Returns:
        mux: Function (subcarrier_signals) -> combined_signal
        demux: Function (combined_signal) -> subcarrier_signals

    Note:
        This is a placeholder implementation.
    '''
    # Sub-carrier frequency offset: ±¼·Fb·(1+α)
    freq_offset = baud_rate * (1 + rolloff) / 4

    def mux(subcarrier_signals):
        '''
        Multiplex sub-carrier signals.

        Args:
            subcarrier_signals: Array of shape (num_subcarriers, num_samples, num_pols)
                                or list of sub-carrier signal arrays

        Returns:
            combined: Combined signal, shape (num_samples, num_pols)
        '''
        # TODO: Implement proper sub-carrier multiplexing:
        # 1. RRC filter each sub-carrier
        # 2. Frequency shift by ±freq_offset
        # 3. Sum the sub-carriers

        # Placeholder: simple sum
        if isinstance(subcarrier_signals, list):
            subcarrier_signals = jnp.stack(subcarrier_signals, axis=0)
        combined = jnp.sum(subcarrier_signals, axis=0)
        return combined

    def demux(combined_signal):
        '''
        Demultiplex combined signal into sub-carriers.

        Args:
            combined: Combined signal, shape (num_samples, num_pols)

        Returns:
            subcarrier_signals: Array of shape (num_subcarriers, num_samples, num_pols)
        '''
        # TODO: Implement proper sub-carrier demultiplexing:
        # 1. Frequency shift to baseband for each sub-carrier
        # 2. Low-pass filter
        # 3. Matched filter (RRC)

        # Placeholder: duplicate signal
        subcarrier_signals = jnp.stack([combined_signal] * num_subcarriers, axis=0)
        return subcarrier_signals

    return mux, demux


def subcarrier_distribute(num_subcarriers=2, interleave_bits=128):
    '''
    Distribute OFEC output to sub-carriers.

    After OFEC interleaving, bits are distributed to sub-carriers in groups
    of 128 bits in round-robin fashion.

    Args:
        num_subcarriers: Number of sub-carriers (default: 2)
        interleave_bits: Bits per distribution group (default: 128)

    Returns:
        distribute: Function (bits) -> (sc0_bits, sc1_bits, ...)
        merge: Function (sc0_bits, sc1_bits, ...) -> bits

    Note:
        This is a placeholder implementation.
    '''

    def distribute(bits):
        '''
        Distribute bits to sub-carriers.

        Args:
            bits: Input bits from OFEC interleaver

        Returns:
            subcarrier_bits: Tuple of bit arrays, one per sub-carrier
        '''
        # Reshape to groups of interleave_bits
        n_groups = len(bits) // interleave_bits
        bits_grouped = bits[:n_groups * interleave_bits].reshape(n_groups, interleave_bits)

        # Round-robin distribution
        subcarrier_bits = []
        for i in range(num_subcarriers):
            sc_bits = bits_grouped[i::num_subcarriers].reshape(-1)
            subcarrier_bits.append(sc_bits)

        return tuple(subcarrier_bits)

    def merge(subcarrier_bits):
        '''
        Merge sub-carrier bits back to single stream.

        Args:
            subcarrier_bits: Tuple of bit arrays from each sub-carrier

        Returns:
            bits: Merged bit stream
        '''
        # Interleave back
        sc_arrays = [sb.reshape(-1, interleave_bits) for sb in subcarrier_bits]
        n_groups = sc_arrays[0].shape[0]

        merged_groups = []
        for i in range(n_groups):
            for sc in sc_arrays:
                merged_groups.append(sc[i])

        bits = jnp.concatenate(merged_groups)
        return bits

    return distribute, merge

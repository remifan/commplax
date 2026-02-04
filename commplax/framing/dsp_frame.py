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

"""DSP frame structure kernels.

DSP framing maps encoded bits to DP-16QAM symbols with overhead insertion.
Used in 1600ZR/ZR+ for the coherent optical interface.

Reference:
    [1] OIF 1600ZR+ Implementation Agreement (oif2024.447.06), Section 6.7
    [2] ITU-T G.709.6, Clause 11
"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple

# =============================================================================
# DSP Frame Parameters
# =============================================================================

# Frame structure
SUBFRAME_SYMBOLS = 7296           # Symbols per DSP sub-frame
SUPERFRAME_SUBFRAMES = 12         # Sub-frames per super-frame (ZR+)
SUPERFRAME_SYMBOLS = SUBFRAME_SYMBOLS * SUPERFRAME_SUBFRAMES  # 87,552

# Payload per super-frame
PAYLOAD_SYMBOLS = 86016           # Data symbols (688,128 bits / 8)
OVERHEAD_SYMBOLS = SUPERFRAME_SYMBOLS - PAYLOAD_SYMBOLS  # 1,536

# Overhead positions
PILOT_INTERVAL = 64               # Pilot symbol every 64 symbols
TS_SYMBOLS = 11                   # Training sequence symbols per sub-frame
FAW_SYMBOLS = 22                  # Frame alignment word (first sub-frame only)
GOI_SYMBOLS = 5                   # Generic optical impairment (first sub-frame)
RES_SYMBOLS = 15                  # Reserved (first sub-frame)
VSU_SYMBOLS = 6                   # Vendor specific use (first sub-frame)

# QPSK amplitude for overhead symbols
QPSK_S_1600ZRP = 3                # Outer QPSK for 1600ZR+
QPSK_S_1200ZRP = 1                # Inner QPSK for 1200ZR+


# =============================================================================
# Symbol Mapping (16QAM)
# =============================================================================

# Gray mapping table: (b0, b1) -> amplitude
# (0,0) -> -3, (0,1) -> -1, (1,1) -> +1, (1,0) -> +3
GRAY_MAP = jnp.array([-3, -1, +3, +1])  # Index = b0*2 + b1


def symbol_mapper_16QAM():
    """
    DP-16QAM symbol mapper kernel.

    Maps 8 bits to one DP-16QAM symbol (4 bits per polarization).
    Bit mapping per OIF spec Section 6.7.2:
        - (c_8i, c_8i+2) -> X polarization I component
        - (c_8i+1, c_8i+3) -> X polarization Q component
        - (c_8i+4, c_8i+6) -> Y polarization I component
        - (c_8i+5, c_8i+7) -> Y polarization Q component

    Gray mapping: (0,0)->-3, (0,1)->-1, (1,1)->+1, (1,0)->+3

    Returns:
        map_fn: Function (bits) -> symbols
        demap_fn: Function (symbols) -> bits (hard decision)
    """

    def _bits_to_amplitude(b0, b1):
        """Map 2 bits to PAM-4 amplitude using Gray code."""
        idx = b0 * 2 + b1
        return GRAY_MAP[idx]

    def _amplitude_to_bits(amp):
        """Map PAM-4 amplitude to 2 bits (hard decision)."""
        # Find closest amplitude level
        idx = jnp.argmin(jnp.abs(GRAY_MAP - amp))
        b0 = idx // 2
        b1 = idx % 2
        return b0, b1

    def map_fn(bits: jnp.ndarray) -> jnp.ndarray:
        """
        Map bits to DP-16QAM symbols.

        Args:
            bits: Input bits, shape (N*8,) where N is number of symbols

        Returns:
            symbols: Complex DP-16QAM symbols, shape (N, 2) for X/Y polarizations
        """
        bits = jnp.atleast_1d(bits)
        n_symbols = bits.shape[0] // 8
        bits = bits[:n_symbols * 8].reshape(n_symbols, 8)

        # Map bits to amplitudes per OIF spec mapping
        # XI: (c_8i, c_8i+2), XQ: (c_8i+1, c_8i+3)
        # YI: (c_8i+4, c_8i+6), YQ: (c_8i+5, c_8i+7)
        xi = _bits_to_amplitude(bits[:, 0], bits[:, 2])
        xq = _bits_to_amplitude(bits[:, 1], bits[:, 3])
        yi = _bits_to_amplitude(bits[:, 4], bits[:, 6])
        yq = _bits_to_amplitude(bits[:, 5], bits[:, 7])

        # Form complex symbols
        x_pol = xi + 1j * xq
        y_pol = yi + 1j * yq
        symbols = jnp.stack([x_pol, y_pol], axis=-1)

        return symbols

    def demap_fn(symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Demap DP-16QAM symbols to bits (hard decision).

        Args:
            symbols: Complex symbols, shape (N, 2) for X/Y polarizations

        Returns:
            bits: Demapped bits, shape (N*8,)
        """
        symbols = jnp.atleast_2d(symbols)
        n_symbols = symbols.shape[0]

        # Extract I/Q components
        xi = jnp.real(symbols[:, 0])
        xq = jnp.imag(symbols[:, 0])
        yi = jnp.real(symbols[:, 1])
        yq = jnp.imag(symbols[:, 1])

        # Hard decision: find closest constellation point
        def demap_component(amp):
            idx = jnp.argmin(jnp.abs(GRAY_MAP[None, :] - amp[:, None]), axis=1)
            b0 = idx // 2
            b1 = idx % 2
            return b0, b1

        xi_b0, xi_b1 = demap_component(xi)
        xq_b0, xq_b1 = demap_component(xq)
        yi_b0, yi_b1 = demap_component(yi)
        yq_b0, yq_b1 = demap_component(yq)

        # Reconstruct bit order: c_8i, c_8i+1, c_8i+2, ..., c_8i+7
        bits = jnp.stack([
            xi_b0, xq_b0, xi_b1, xq_b1,
            yi_b0, yq_b0, yi_b1, yq_b1
        ], axis=-1).reshape(-1)

        return bits

    return map_fn, demap_fn


# =============================================================================
# Training and Pilot Sequences
# =============================================================================

def _generate_training_sequence(S: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate training sequence per OIF spec Table 19.

    Args:
        S: QPSK amplitude (3 for 1600ZR+, 1 for 1200ZR+)

    Returns:
        ts_x: Training sequence for X polarization (11 QPSK symbols)
        ts_y: Training sequence for Y polarization (11 QPSK symbols)
    """
    # From Table 19 of OIF spec (first 11 symbols)
    # Format: complex value = I + jQ where I,Q in {-S, +S}
    ts_x = jnp.array([
        -S + S*1j,   # Index 1
         S + S*1j,   # Index 2
        -S + S*1j,   # Index 3
         S + S*1j,   # Index 4
        -S - S*1j,   # Index 5
         S + S*1j,   # Index 6
        -S + S*1j,   # Index 7
         S - S*1j,   # Index 8
        -S - S*1j,   # Index 9
         S + S*1j,   # Index 10
        -S - S*1j,   # Index 11
    ], dtype=jnp.complex64)

    ts_y = jnp.array([
        -S - S*1j,   # Index 1
        -S - S*1j,   # Index 2
         S - S*1j,   # Index 3
        -S + S*1j,   # Index 4
        -S + S*1j,   # Index 5
         S + S*1j,   # Index 6
         S - S*1j,   # Index 7
         S + S*1j,   # Index 8
        -S + S*1j,   # Index 9
        -S - S*1j,   # Index 10
         S + S*1j,   # Index 11
    ], dtype=jnp.complex64)

    return ts_x, ts_y


def _generate_pilot_sequence(length: int, S: int = 3, seed: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate pilot sequence (PRBS-based QPSK).

    Args:
        length: Number of pilot symbols needed
        S: QPSK amplitude
        seed: Random seed for PRBS

    Returns:
        pilot_x, pilot_y: Pilot sequences for X/Y polarizations
    """
    # Simplified: use deterministic QPSK sequence
    # Full implementation would use PRBS as specified in ITU-T G.709.6
    qpsk_points = jnp.array([
        -S - S*1j, -S + S*1j, S - S*1j, S + S*1j
    ], dtype=jnp.complex64)

    # Generate pseudo-random indices
    idx_x = jnp.arange(length) % 4
    idx_y = (jnp.arange(length) + 2) % 4  # Different sequence for Y

    pilot_x = qpsk_points[idx_x]
    pilot_y = qpsk_points[idx_y]

    return pilot_x, pilot_y


# =============================================================================
# DSP Sub-frame
# =============================================================================

def DSP_subframe(subframe_symbols: int = SUBFRAME_SYMBOLS, mode: str = '1600ZR+'):
    """
    DSP sub-frame kernel.

    A DSP sub-frame contains 7,296 DP-16QAM symbols with overhead:
    - Training sequence (TS): 11 QPSK symbols at start
    - Pilot symbols (PS): Every 64 symbols
    - First sub-frame has additional FAW, GOI, RES, VSU

    Args:
        subframe_symbols: Symbols per sub-frame (default: 7296)
        mode: '1600ZR+' or '1200ZR+' (affects QPSK amplitude)

    Returns:
        frame: Function (data_symbols, subframe_index) -> subframe_with_oh
        deframe: Function (subframe_with_oh, subframe_index) -> data_symbols
    """
    S = QPSK_S_1600ZRP if mode == '1600ZR+' else QPSK_S_1200ZRP

    # Generate overhead sequences
    ts_x, ts_y = _generate_training_sequence(S)
    n_pilots = subframe_symbols // PILOT_INTERVAL
    pilot_x, pilot_y = _generate_pilot_sequence(n_pilots, S)

    # Calculate data capacity per sub-frame
    # Pilots: every 64 symbols = 114 pilots per sub-frame
    # TS: 11 symbols (first symbol is also a pilot)
    data_per_subframe = subframe_symbols - n_pilots - (TS_SYMBOLS - 1)

    def frame(data_symbols: jnp.ndarray, subframe_index: int = 0) -> jnp.ndarray:
        """
        Insert overhead into data symbols to form sub-frame.

        Args:
            data_symbols: Data symbols, shape (N, 2) for X/Y polarizations
            subframe_index: Position within super-frame (0-11)

        Returns:
            subframe: Sub-frame with overhead, shape (7296, 2)
        """
        data_symbols = jnp.atleast_2d(data_symbols)

        # Build sub-frame with overhead insertion
        subframe_x = []
        subframe_y = []
        data_idx = 0

        for i in range(subframe_symbols):
            if i < TS_SYMBOLS:
                # Training sequence at start
                subframe_x.append(ts_x[i])
                subframe_y.append(ts_y[i])
            elif i % PILOT_INTERVAL == 0:
                # Pilot symbol
                pilot_idx = i // PILOT_INTERVAL
                subframe_x.append(pilot_x[pilot_idx % len(pilot_x)])
                subframe_y.append(pilot_y[pilot_idx % len(pilot_y)])
            else:
                # Data symbol
                if data_idx < data_symbols.shape[0]:
                    subframe_x.append(data_symbols[data_idx, 0])
                    subframe_y.append(data_symbols[data_idx, 1])
                    data_idx += 1
                else:
                    # Pad with zeros if not enough data
                    subframe_x.append(0.0 + 0.0j)
                    subframe_y.append(0.0 + 0.0j)

        subframe = jnp.stack([
            jnp.array(subframe_x, dtype=jnp.complex64),
            jnp.array(subframe_y, dtype=jnp.complex64)
        ], axis=-1)

        return subframe

    def deframe(subframe: jnp.ndarray, subframe_index: int = 0) -> jnp.ndarray:
        """
        Extract data symbols from sub-frame.

        Args:
            subframe: Sub-frame with overhead, shape (7296, 2)
            subframe_index: Position within super-frame (0-11)

        Returns:
            data_symbols: Extracted data symbols, shape (N, 2)
        """
        subframe = jnp.atleast_2d(subframe)

        # Extract data symbols (skip TS and pilots)
        data_x = []
        data_y = []

        for i in range(subframe.shape[0]):
            if i < TS_SYMBOLS:
                continue  # Skip training sequence
            elif i % PILOT_INTERVAL == 0:
                continue  # Skip pilot
            else:
                data_x.append(subframe[i, 0])
                data_y.append(subframe[i, 1])

        data_symbols = jnp.stack([
            jnp.array(data_x, dtype=jnp.complex64),
            jnp.array(data_y, dtype=jnp.complex64)
        ], axis=-1)

        return data_symbols

    return frame, deframe


# =============================================================================
# DSP Super-frame
# =============================================================================

def DSP_superframe(num_subframes: int = SUPERFRAME_SUBFRAMES, mode: str = '1600ZR+'):
    """
    DSP super-frame kernel.

    A DSP super-frame contains 12 sequential sub-frames (87,552 symbols total).
    The first sub-frame includes FAW, GOI, RES, and VSU overhead.

    Args:
        num_subframes: Number of sub-frames per super-frame (default: 12)
        mode: '1600ZR+' or '1200ZR+' (affects QPSK amplitude)

    Returns:
        frame: Function (data_symbols) -> superframe
        deframe: Function (superframe) -> data_symbols
    """
    subframe_frame, subframe_deframe = DSP_subframe(mode=mode)

    def frame(data_symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble super-frame from data symbols.

        Args:
            data_symbols: Data symbols, shape (N, 2) for X/Y polarizations

        Returns:
            superframe: Complete super-frame, shape (87552, 2)
        """
        data_symbols = jnp.atleast_2d(data_symbols)
        n_total = data_symbols.shape[0]
        n_per_subframe = n_total // num_subframes

        subframes = []
        for i in range(num_subframes):
            start = i * n_per_subframe
            end = start + n_per_subframe
            sf_data = data_symbols[start:end]
            sf = subframe_frame(sf_data, subframe_index=i)
            subframes.append(sf)

        superframe = jnp.concatenate(subframes, axis=0)
        return superframe

    def deframe(superframe: jnp.ndarray) -> jnp.ndarray:
        """
        Extract data symbols from super-frame.

        Args:
            superframe: Complete super-frame, shape (87552, 2)

        Returns:
            data_symbols: Extracted data symbols
        """
        superframe = jnp.atleast_2d(superframe)
        sf_len = superframe.shape[0] // num_subframes

        data_symbols = []
        for i in range(num_subframes):
            start = i * sf_len
            end = start + sf_len
            sf = superframe[start:end]
            data = subframe_deframe(sf, subframe_index=i)
            data_symbols.append(data)

        return jnp.concatenate(data_symbols, axis=0)

    return frame, deframe


# =============================================================================
# Interleaver Merge and Sub-carrier Distribution
# =============================================================================

def interleaver_merge(num_interleavers: int = 4):
    """
    Merge OFEC interleaver outputs for symbol mapping.

    Per OIF spec Section 6.7.1:
    - Merge 64 bits from each of 4 interleavers in round-robin
    - Results in 256-bit groups (32 symbols worth)

    Args:
        num_interleavers: Number of OFEC interleavers (default: 4)

    Returns:
        merge: Function (interleaver_outputs) -> merged_bits
        split: Function (merged_bits) -> interleaver_outputs
    """
    bits_per_group = 64  # 64 bits = 8 symbols per interleaver per group

    def merge(interleaver_outputs: list) -> jnp.ndarray:
        """
        Merge interleaver outputs into single stream.

        Args:
            interleaver_outputs: List of 4 bit arrays from interleavers

        Returns:
            merged: Merged bit stream for symbol mapping
        """
        # Ensure all have same length
        min_len = min(out.shape[0] for out in interleaver_outputs)
        n_groups = min_len // bits_per_group

        merged = []
        for g in range(n_groups):
            for il in range(num_interleavers):
                start = g * bits_per_group
                end = start + bits_per_group
                merged.append(interleaver_outputs[il][start:end])

        return jnp.concatenate(merged)

    def split(merged_bits: jnp.ndarray) -> list:
        """
        Split merged stream back to interleaver outputs.

        Args:
            merged_bits: Merged bit stream

        Returns:
            interleaver_outputs: List of 4 bit arrays
        """
        n_total = merged_bits.shape[0]
        group_size = bits_per_group * num_interleavers
        n_groups = n_total // group_size

        outputs = [[] for _ in range(num_interleavers)]

        for g in range(n_groups):
            for il in range(num_interleavers):
                start = g * group_size + il * bits_per_group
                end = start + bits_per_group
                outputs[il].append(merged_bits[start:end])

        return [jnp.concatenate(out) for out in outputs]

    return merge, split


def subcarrier_bit_distribute(num_subcarriers: int = 2):
    """
    Distribute merged bits to sub-carriers.

    Per OIF spec Section 6.7.1:
    - Distribute 256-bit groups to 2 sub-carriers (128 bits each)
    - First 128 bits -> sub-carrier 0, next 128 bits -> sub-carrier 1

    Args:
        num_subcarriers: Number of sub-carriers (default: 2)

    Returns:
        distribute: Function (merged_bits) -> (sc0_bits, sc1_bits)
        combine: Function (sc0_bits, sc1_bits) -> merged_bits
    """
    bits_per_sc = 128  # 128 bits per sub-carrier per group

    def distribute(merged_bits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Distribute merged bits to sub-carriers."""
        n_total = merged_bits.shape[0]
        group_size = bits_per_sc * num_subcarriers
        n_groups = n_total // group_size

        sc_bits = [[] for _ in range(num_subcarriers)]

        for g in range(n_groups):
            for sc in range(num_subcarriers):
                start = g * group_size + sc * bits_per_sc
                end = start + bits_per_sc
                sc_bits[sc].append(merged_bits[start:end])

        return tuple(jnp.concatenate(sc) for sc in sc_bits)

    def combine(sc_bits: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Combine sub-carrier bits back to single stream."""
        n_groups = sc_bits[0].shape[0] // bits_per_sc

        merged = []
        for g in range(n_groups):
            for sc in range(num_subcarriers):
                start = g * bits_per_sc
                end = start + bits_per_sc
                merged.append(sc_bits[sc][start:end])

        return jnp.concatenate(merged)

    return distribute, combine

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

"""DSP frame structure kernels for coherent optical systems.

Supports multiple OIF specifications:
- 400ZR (OIF-400ZR-03.0)
- 800ZR (OIF-800ZR-01.0)
- 1600ZR+ (OIF 1600ZR+ IA)

DSP framing maps encoded bits to DP-16QAM symbols with overhead insertion
including training sequences, pilots, and frame alignment words.

Reference:
    [1] OIF-400ZR-03.0.1, Section 12 (DSP framing)
    [2] OIF 1600ZR+ Implementation Agreement, Section 6.7
"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Optional
import dataclasses as dc


# =============================================================================
# Spec Configurations
# =============================================================================

@dc.dataclass(frozen=True)
class DSPFrameConfig:
    """DSP frame parameters for a specific OIF spec."""
    name: str
    subframe_symbols: int       # Symbols per DSP sub-frame
    superframe_subframes: int   # Sub-frames per super-frame
    pilot_interval: int         # Pilot symbol interval
    ts_symbols: int             # Training sequence length
    faw_symbols: int            # Frame alignment word (first sub-frame)
    reserved_symbols: int       # Reserved symbols (first sub-frame)
    qpsk_amplitude: int         # QPSK amplitude for overhead (±S)
    pilot_seed_x: int           # PRBS seed for X polarization
    pilot_seed_y: int           # PRBS seed for Y polarization
    pilot_poly: int             # PRBS generator polynomial

    @property
    def superframe_symbols(self) -> int:
        return self.subframe_symbols * self.superframe_subframes

    @property
    def pilots_per_subframe(self) -> int:
        return self.subframe_symbols // self.pilot_interval


# 400ZR configuration (OIF-400ZR-03.0.1)
CONFIG_400ZR = DSPFrameConfig(
    name='400ZR',
    subframe_symbols=3712,
    superframe_subframes=49,
    pilot_interval=32,
    ts_symbols=11,
    faw_symbols=22,
    reserved_symbols=76,
    qpsk_amplitude=3,
    pilot_seed_x=0x19E,
    pilot_seed_y=0x0D0,
    pilot_poly=0b10100011001,  # x^10 + x^8 + x^4 + x^3 + 1
)

# 800ZR configuration (OIF-800ZR-01.0) - placeholder, verify from spec
CONFIG_800ZR = DSPFrameConfig(
    name='800ZR',
    subframe_symbols=3712,      # TBD - verify from spec
    superframe_subframes=49,    # TBD - verify from spec
    pilot_interval=32,
    ts_symbols=11,
    faw_symbols=22,
    reserved_symbols=76,
    qpsk_amplitude=3,
    pilot_seed_x=0x19E,
    pilot_seed_y=0x0D0,
    pilot_poly=0b10100011001,
)

# 1600ZR+ configuration (OIF 1600ZR+ IA)
CONFIG_1600ZRP = DSPFrameConfig(
    name='1600ZR+',
    subframe_symbols=7296,
    superframe_subframes=12,
    pilot_interval=64,
    ts_symbols=11,
    faw_symbols=22,
    reserved_symbols=48,        # GOI(5) + RES(15) + VSU(6) + other
    qpsk_amplitude=3,
    pilot_seed_x=0x000,         # TBD - verify from spec
    pilot_seed_y=0x000,
    pilot_poly=0b10000000001,   # TBD - verify from spec
)

# Lookup by name
CONFIGS = {
    '400ZR': CONFIG_400ZR,
    '800ZR': CONFIG_800ZR,
    '1600ZR+': CONFIG_1600ZRP,
    '1600ZRP': CONFIG_1600ZRP,
}


def get_config(spec: str) -> DSPFrameConfig:
    """Get configuration for a spec by name."""
    if spec not in CONFIGS:
        raise ValueError(f"Unknown spec: {spec}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[spec]


# =============================================================================
# Symbol Mapping (16QAM) - Common to all specs
# =============================================================================

# Gray mapping table: (b0, b1) -> amplitude
# 400ZR: (0,0)->-3, (0,1)->-1, (1,0)->+3, (1,1)->+1
GRAY_MAP_400ZR = jnp.array([-3, -1, +3, +1])  # Index = b0*2 + b1

# 1600ZR+ uses same Gray mapping
GRAY_MAP = GRAY_MAP_400ZR


def symbol_mapper_16QAM(spec: str = '400ZR'):
    """
    DP-16QAM symbol mapper kernel.

    Maps 8 bits to one DP-16QAM symbol (4 bits per polarization).

    Bit mapping (400ZR spec Section 11):
        For codeword bits c[0:127] mapping to 16 symbols s[0:15]:
        - s[i] X-pol I: (c[8i], c[8i+1])
        - s[i] X-pol Q: (c[8i+2], c[8i+3])
        - s[i] Y-pol I: (c[8i+4], c[8i+5])
        - s[i] Y-pol Q: (c[8i+6], c[8i+7])

    Gray mapping: (0,0)->-3, (0,1)->-1, (1,0)->+3, (1,1)->+1

    Args:
        spec: OIF spec name ('400ZR', '800ZR', '1600ZR+')

    Returns:
        map_fn: Function (bits) -> symbols
        demap_fn: Function (symbols) -> bits (hard decision)
    """
    gray = GRAY_MAP

    def map_fn(bits: jnp.ndarray) -> jnp.ndarray:
        """
        Map bits to DP-16QAM symbols.

        Args:
            bits: Input bits, shape (N*8,) where N is number of symbols

        Returns:
            symbols: Complex DP-16QAM symbols, shape (N, 2) for X/Y polarizations
        """
        bits = jnp.atleast_1d(bits).astype(jnp.int32)
        n_symbols = bits.shape[0] // 8
        bits = bits[:n_symbols * 8].reshape(n_symbols, 8)

        # Map bit pairs to amplitudes using Gray code
        # Index = b0*2 + b1
        xi = gray[bits[:, 0] * 2 + bits[:, 1]]
        xq = gray[bits[:, 2] * 2 + bits[:, 3]]
        yi = gray[bits[:, 4] * 2 + bits[:, 5]]
        yq = gray[bits[:, 6] * 2 + bits[:, 7]]

        # Form complex symbols
        x_pol = xi + 1j * xq
        y_pol = yi + 1j * yq
        symbols = jnp.stack([x_pol, y_pol], axis=-1)

        return symbols.astype(jnp.complex64)

    def demap_fn(symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Demap DP-16QAM symbols to bits (hard decision).

        Args:
            symbols: Complex symbols, shape (N, 2) for X/Y polarizations

        Returns:
            bits: Demapped bits, shape (N*8,)
        """
        symbols = jnp.atleast_2d(symbols)

        # Extract I/Q components
        xi = jnp.real(symbols[:, 0])
        xq = jnp.imag(symbols[:, 0])
        yi = jnp.real(symbols[:, 1])
        yq = jnp.imag(symbols[:, 1])

        # Hard decision: find closest constellation point
        def demap_component(amp):
            idx = jnp.argmin(jnp.abs(gray[None, :] - amp[:, None]), axis=1)
            b0 = idx // 2
            b1 = idx % 2
            return b0, b1

        xi_b0, xi_b1 = demap_component(xi)
        xq_b0, xq_b1 = demap_component(xq)
        yi_b0, yi_b1 = demap_component(yi)
        yq_b0, yq_b1 = demap_component(yq)

        # Reconstruct bit order
        bits = jnp.stack([
            xi_b0, xi_b1, xq_b0, xq_b1,
            yi_b0, yi_b1, yq_b0, yq_b1
        ], axis=-1).reshape(-1)

        return bits.astype(jnp.uint8)

    return map_fn, demap_fn


# =============================================================================
# Training Sequences
# =============================================================================

def _training_sequence_400ZR(S: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate 400ZR training sequence per OIF-400ZR-03.0.1 Table 14.

    Args:
        S: QPSK amplitude (3 for standard)

    Returns:
        ts_x: Training sequence for X polarization (11 QPSK symbols)
        ts_y: Training sequence for Y polarization (11 QPSK symbols)
    """
    # Table 14: Training symbol sequence
    # First symbol (*) is also processed as pilot
    ts_x = jnp.array([
        -S + S*1j,   # Index 1*
         S + S*1j,   # Index 2
        -S + S*1j,   # Index 3
         S + S*1j,   # Index 4
        -S - S*1j,   # Index 5
         S + S*1j,   # Index 6
        -S - S*1j,   # Index 7
        -S - S*1j,   # Index 8
         S + S*1j,   # Index 9
         S - S*1j,   # Index 10
         S - S*1j,   # Index 11
    ], dtype=jnp.complex64)

    ts_y = jnp.array([
        -S - S*1j,   # Index 1*
        -S - S*1j,   # Index 2
         S - S*1j,   # Index 3
        -S + S*1j,   # Index 4
        -S + S*1j,   # Index 5
         S + S*1j,   # Index 6
        -S - S*1j,   # Index 7
        -S + S*1j,   # Index 8
         S - S*1j,   # Index 9
         S + S*1j,   # Index 10
         S - S*1j,   # Index 11
    ], dtype=jnp.complex64)

    return ts_x, ts_y


def _training_sequence_1600ZRP(S: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate 1600ZR+ training sequence per OIF spec Table 19.

    Args:
        S: QPSK amplitude (3 for 1600ZR+, 1 for 1200ZR+)

    Returns:
        ts_x: Training sequence for X polarization (11 QPSK symbols)
        ts_y: Training sequence for Y polarization (11 QPSK symbols)
    """
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


def get_training_sequence(config: DSPFrameConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get training sequence for a given spec configuration."""
    S = config.qpsk_amplitude
    if config.name == '400ZR' or config.name == '800ZR':
        return _training_sequence_400ZR(S)
    else:
        return _training_sequence_1600ZRP(S)


# =============================================================================
# Frame Alignment Word (FAW)
# =============================================================================

def _faw_400ZR(S: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate 400ZR Frame Alignment Word per OIF-400ZR-03.0.1 Table 13.

    Args:
        S: QPSK amplitude

    Returns:
        faw_x: FAW for X polarization (22 QPSK symbols)
        faw_y: FAW for Y polarization (22 QPSK symbols)
    """
    # Table 13: FAW sequence (22 symbols)
    faw_x = jnp.array([
         S - S*1j,   # 1
         S + S*1j,   # 2
         S + S*1j,   # 3
         S + S*1j,   # 4
         S - S*1j,   # 5
         S - S*1j,   # 6
        -S - S*1j,   # 7
         S + S*1j,   # 8
        -S - S*1j,   # 9
        -S + S*1j,   # 10
        -S + S*1j,   # 11
         S - S*1j,   # 12
        -S - S*1j,   # 13
        -S - S*1j,   # 14
        -S + S*1j,   # 15
         S + S*1j,   # 16
        -S - S*1j,   # 17
         S - S*1j,   # 18
        -S + S*1j,   # 19
         S + S*1j,   # 20
        -S - S*1j,   # 21
        -S + S*1j,   # 22
    ], dtype=jnp.complex64)

    faw_y = jnp.array([
         S + S*1j,   # 1
        -S + S*1j,   # 2
        -S - S*1j,   # 3
        -S + S*1j,   # 4
         S - S*1j,   # 5
         S + S*1j,   # 6
         S - S*1j,   # 7
         S - S*1j,   # 8
        -S - S*1j,   # 9
         S - S*1j,   # 10
         S + S*1j,   # 11
        -S + S*1j,   # 12
        -S + S*1j,   # 13
         S + S*1j,   # 14
        -S - S*1j,   # 15
         S + S*1j,   # 16
        -S - S*1j,   # 17
        -S + S*1j,   # 18
         S - S*1j,   # 19
        -S - S*1j,   # 20
         S - S*1j,   # 21
        -S + S*1j,   # 22
    ], dtype=jnp.complex64)

    return faw_x, faw_y


def get_faw(config: DSPFrameConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get Frame Alignment Word for a given spec configuration."""
    S = config.qpsk_amplitude
    # 400ZR and 800ZR use same FAW structure
    return _faw_400ZR(S)


# =============================================================================
# Pilot Sequence (PRBS-based)
# =============================================================================

def _prbs_step(state: int, poly: int, width: int) -> Tuple[int, int]:
    """Single PRBS step: returns (new_state, output_bit)."""
    # XOR feedback based on polynomial taps
    feedback = 0
    temp = state & poly
    while temp:
        feedback ^= (temp & 1)
        temp >>= 1
    # Shift and insert feedback
    output = state & 1
    new_state = (state >> 1) | (feedback << (width - 1))
    return new_state, output


def generate_pilot_sequence(config: DSPFrameConfig, length: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate pilot sequence using PRBS mapped to QPSK.

    Per 400ZR spec Section 12.3:
    - PRBS10 with polynomial x^10 + x^8 + x^4 + x^3 + 1
    - Different seeds for X (0x19E) and Y (0x0D0)
    - 2 bits -> 1 QPSK symbol

    Args:
        config: DSP frame configuration
        length: Number of pilot symbols needed

    Returns:
        pilot_x, pilot_y: Pilot sequences for X/Y polarizations
    """
    S = config.qpsk_amplitude
    poly = config.pilot_poly
    width = 10  # PRBS10

    # QPSK mapping: 2 bits -> complex symbol
    # 00 -> -S-Sj, 01 -> -S+Sj, 10 -> +S-Sj, 11 -> +S+Sj
    qpsk_map = jnp.array([
        -S - S*1j,  # 00
        -S + S*1j,  # 01
         S - S*1j,  # 10
         S + S*1j,  # 11
    ], dtype=jnp.complex64)

    def generate_one(seed: int) -> jnp.ndarray:
        """Generate pilot sequence from a seed."""
        state = seed
        symbols = []
        for _ in range(length):
            # Get 2 bits for one QPSK symbol
            state, b0 = _prbs_step(state, poly, width)
            state, b1 = _prbs_step(state, poly, width)
            idx = b0 * 2 + b1
            symbols.append(qpsk_map[idx])
        return jnp.array(symbols, dtype=jnp.complex64)

    pilot_x = generate_one(config.pilot_seed_x)
    pilot_y = generate_one(config.pilot_seed_y)

    return pilot_x, pilot_y


# =============================================================================
# DSP Sub-frame
# =============================================================================

def DSP_subframe(spec: str = '400ZR'):
    """
    DSP sub-frame kernel.

    Structure (400ZR example with 3712 symbols, pilot every 32):
    - First sub-frame: TS(11) + FAW(22) + RES(76) + data/pilots
    - Other sub-frames: TS(11) + data/pilots
    - Pilot at every pilot_interval symbols (including first TS symbol)

    Args:
        spec: OIF spec name ('400ZR', '800ZR', '1600ZR+')

    Returns:
        frame: Function (data_symbols, subframe_index) -> subframe_with_oh
        deframe: Function (subframe_with_oh, subframe_index) -> data_symbols
    """
    config = get_config(spec)
    S = config.qpsk_amplitude

    # Generate overhead sequences
    ts_x, ts_y = get_training_sequence(config)
    faw_x, faw_y = get_faw(config)
    n_pilots = config.pilots_per_subframe
    pilot_x, pilot_y = generate_pilot_sequence(config, n_pilots)

    # Calculate data capacity
    # First sub-frame: TS + FAW + RES + (pilots - overlap with TS[0])
    # Other sub-frames: TS + (pilots - overlap with TS[0])
    first_overhead = config.ts_symbols + config.faw_symbols + config.reserved_symbols
    other_overhead = config.ts_symbols

    # Pilots overlap with first TS symbol
    pilots_effective = n_pilots - 1  # First pilot is TS[0]

    data_first = config.subframe_symbols - first_overhead - pilots_effective
    data_other = config.subframe_symbols - other_overhead - pilots_effective

    def frame(data_symbols: jnp.ndarray, subframe_index: int = 0) -> jnp.ndarray:
        """
        Insert overhead into data symbols to form sub-frame.

        Args:
            data_symbols: Data symbols, shape (N, 2) for X/Y polarizations
            subframe_index: Position within super-frame (0 = first)

        Returns:
            subframe: Sub-frame with overhead, shape (subframe_symbols, 2)
        """
        data_symbols = jnp.atleast_2d(data_symbols)
        is_first = (subframe_index == 0)

        # Pre-allocate output
        out_x = jnp.zeros(config.subframe_symbols, dtype=jnp.complex64)
        out_y = jnp.zeros(config.subframe_symbols, dtype=jnp.complex64)

        # Build sub-frame symbol by symbol
        data_idx = 0
        pilot_idx = 0

        result_x = []
        result_y = []

        for i in range(config.subframe_symbols):
            is_pilot_position = (i % config.pilot_interval == 0)

            if i < config.ts_symbols:
                # Training sequence (first TS symbol is also pilot)
                result_x.append(ts_x[i])
                result_y.append(ts_y[i])
                if i == 0:
                    pilot_idx += 1  # TS[0] counts as pilot
            elif is_first and i < config.ts_symbols + config.faw_symbols:
                # FAW (first sub-frame only)
                faw_i = i - config.ts_symbols
                result_x.append(faw_x[faw_i])
                result_y.append(faw_y[faw_i])
            elif is_first and i < first_overhead:
                # Reserved symbols (first sub-frame only) - use random QPSK
                # Per spec: "should be randomized to avoid strong tones"
                res_i = i - config.ts_symbols - config.faw_symbols
                result_x.append(pilot_x[pilot_idx % len(pilot_x)])
                result_y.append(pilot_y[pilot_idx % len(pilot_y)])
            elif is_pilot_position:
                # Pilot symbol
                result_x.append(pilot_x[pilot_idx % len(pilot_x)])
                result_y.append(pilot_y[pilot_idx % len(pilot_y)])
                pilot_idx += 1
            else:
                # Data symbol
                if data_idx < data_symbols.shape[0]:
                    result_x.append(data_symbols[data_idx, 0])
                    result_y.append(data_symbols[data_idx, 1])
                    data_idx += 1
                else:
                    result_x.append(0.0 + 0.0j)
                    result_y.append(0.0 + 0.0j)

        subframe = jnp.stack([
            jnp.array(result_x, dtype=jnp.complex64),
            jnp.array(result_y, dtype=jnp.complex64)
        ], axis=-1)

        return subframe

    def deframe(subframe: jnp.ndarray, subframe_index: int = 0) -> jnp.ndarray:
        """
        Extract data symbols from sub-frame.

        Args:
            subframe: Sub-frame with overhead, shape (subframe_symbols, 2)
            subframe_index: Position within super-frame (0 = first)

        Returns:
            data_symbols: Extracted data symbols, shape (N, 2)
        """
        subframe = jnp.atleast_2d(subframe)
        is_first = (subframe_index == 0)
        first_oh = first_overhead if is_first else other_overhead

        data_x = []
        data_y = []

        for i in range(subframe.shape[0]):
            is_pilot_position = (i % config.pilot_interval == 0)

            if i < config.ts_symbols:
                continue  # Skip TS
            elif is_first and i < first_oh:
                continue  # Skip FAW + RES
            elif is_pilot_position:
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

def DSP_superframe(spec: str = '400ZR'):
    """
    DSP super-frame kernel.

    A DSP super-frame contains multiple sequential sub-frames:
    - 400ZR: 49 sub-frames × 3712 symbols = 181,888 symbols
    - 1600ZR+: 12 sub-frames × 7296 symbols = 87,552 symbols

    Args:
        spec: OIF spec name ('400ZR', '800ZR', '1600ZR+')

    Returns:
        frame: Function (data_symbols) -> superframe
        deframe: Function (superframe) -> data_symbols
    """
    config = get_config(spec)
    subframe_frame, subframe_deframe = DSP_subframe(spec)

    def frame(data_symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble super-frame from data symbols.

        Args:
            data_symbols: Data symbols, shape (N, 2) for X/Y polarizations

        Returns:
            superframe: Complete super-frame, shape (superframe_symbols, 2)
        """
        data_symbols = jnp.atleast_2d(data_symbols)
        n_total = data_symbols.shape[0]
        n_per_subframe = n_total // config.superframe_subframes

        subframes = []
        for i in range(config.superframe_subframes):
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
            superframe: Complete super-frame, shape (superframe_symbols, 2)

        Returns:
            data_symbols: Extracted data symbols
        """
        superframe = jnp.atleast_2d(superframe)
        sf_len = config.subframe_symbols

        data_symbols = []
        for i in range(config.superframe_subframes):
            start = i * sf_len
            end = start + sf_len
            sf = superframe[start:end]
            data = subframe_deframe(sf, subframe_index=i)
            data_symbols.append(data)

        return jnp.concatenate(data_symbols, axis=0)

    return frame, deframe


# =============================================================================
# Convenience exports for backward compatibility
# =============================================================================

# 400ZR parameters
SUBFRAME_SYMBOLS_400ZR = CONFIG_400ZR.subframe_symbols
SUPERFRAME_SUBFRAMES_400ZR = CONFIG_400ZR.superframe_subframes
SUPERFRAME_SYMBOLS_400ZR = CONFIG_400ZR.superframe_symbols
PILOT_INTERVAL_400ZR = CONFIG_400ZR.pilot_interval
TS_SYMBOLS_400ZR = CONFIG_400ZR.ts_symbols

# 1600ZR+ parameters (backward compat)
SUBFRAME_SYMBOLS = CONFIG_1600ZRP.subframe_symbols
SUPERFRAME_SUBFRAMES = CONFIG_1600ZRP.superframe_subframes
SUPERFRAME_SYMBOLS = CONFIG_1600ZRP.superframe_symbols
PILOT_INTERVAL = CONFIG_1600ZRP.pilot_interval
TS_SYMBOLS = CONFIG_1600ZRP.ts_symbols
FAW_SYMBOLS = CONFIG_1600ZRP.faw_symbols
PAYLOAD_SYMBOLS = 86016  # Legacy


# =============================================================================
# Interleaver Merge (1600ZR+ specific, kept for compatibility)
# =============================================================================

def interleaver_merge(num_interleavers: int = 4):
    """
    Merge OFEC interleaver outputs for symbol mapping (1600ZR+ specific).

    Per OIF spec Section 6.7.1:
    - Merge 64 bits from each of 4 interleavers in round-robin
    - Results in 256-bit groups (32 symbols worth)

    Args:
        num_interleavers: Number of OFEC interleavers (default: 4)

    Returns:
        merge: Function (interleaver_outputs) -> merged_bits
        split: Function (merged_bits) -> interleaver_outputs
    """
    bits_per_group = 64

    def merge(interleaver_outputs: list) -> jnp.ndarray:
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
    Distribute merged bits to sub-carriers (1600ZR+ specific).

    Per OIF spec Section 6.7.1:
    - Distribute 256-bit groups to sub-carriers (128 bits each)

    Args:
        num_subcarriers: Number of sub-carriers (default: 2)

    Returns:
        distribute: Function (merged_bits) -> tuple of bit arrays
        combine: Function (tuple of bit arrays) -> merged_bits
    """
    bits_per_sc = 128

    def distribute(merged_bits: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
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
        n_groups = sc_bits[0].shape[0] // bits_per_sc

        merged = []
        for g in range(n_groups):
            for sc in range(len(sc_bits)):
                start = g * bits_per_sc
                end = start + bits_per_sc
                merged.append(sc_bits[sc][start:end])

        return jnp.concatenate(merged)

    return distribute, combine

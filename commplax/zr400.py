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

"""400ZR TX/RX chain integration.

This module integrates the 400ZR signal processing components into
complete TX and RX chains per OIF-400ZR-03.0.1.

TX Chain (encoding):
    bits → [Staircase FEC] → Scrambler → Conv Interleaver → Hamming(128,119)
        → Symbol Interleave → DP-16QAM Mapping → DSP Framing → symbols

RX Chain (decoding):
    symbols → DSP Deframing → DP-16QAM Demapping → Symbol Deinterleave
        → Hamming Decode → Conv Deinterleave → Descrambler → [Staircase FEC] → bits

Note:
    Staircase FEC (outer code) is not yet implemented. The current chain
    operates on post-staircase bits for TX and pre-staircase bits for RX.

Reference:
    [1] OIF-400ZR-03.0.1, Sections 10-12
"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Optional
import dataclasses as dc

from commplax.fec import (
    Hamming128_119,
    Hamming128_119_soft,
    ConvolutionalInterleaver,
    FrameScrambler,
    HAMMING_N,
    HAMMING_K,
    CI_DEPTH,
)
from commplax.framing import (
    DSP_subframe,
    DSP_superframe,
    symbol_mapper_16QAM,
    get_config,
    CONFIG_400ZR,
)


# =============================================================================
# 400ZR Frame Parameters
# =============================================================================

@dc.dataclass(frozen=True)
class ZR400Config:
    """400ZR system configuration."""
    # FEC
    hamming_k: int = 119          # Hamming message bits
    hamming_n: int = 128          # Hamming codeword bits
    ci_depth: int = 16            # Conv interleaver depth
    ci_blocks_per_frame: int = 10976  # 119-bit blocks per frame

    # Symbol mapping
    bits_per_symbol: int = 8      # DP-16QAM: 4 bits × 2 polarizations
    symbol_interleave_depth: int = 8  # 8-way symbol interleave

    # DSP framing
    subframe_symbols: int = 3712
    superframe_subframes: int = 49
    pilot_interval: int = 32

    @property
    def superframe_symbols(self) -> int:
        return self.subframe_symbols * self.superframe_subframes  # 181,888

    @property
    def hamming_codewords_per_frame(self) -> int:
        return self.ci_blocks_per_frame  # 10,976

    @property
    def symbols_per_frame(self) -> int:
        # 10976 codewords × 128 bits / 8 bits per symbol = 175,616 data symbols
        return (self.hamming_codewords_per_frame * self.hamming_n) // self.bits_per_symbol


CONFIG = ZR400Config()


# =============================================================================
# Symbol Interleaver (8-way)
# =============================================================================

def SymbolInterleaver(depth: int = 8):
    """
    8-way symbol interleaver per OIF-400ZR-03.0.1 Section 11.1.

    Interleaves symbols from 8 consecutive Hamming codewords to
    de-correlate noise and distribute symbols between pilots.

    Each Hamming codeword (128 bits) maps to 16 DP-16QAM symbols.
    The interleaver takes 8 codewords (128 symbols) and interleaves them.

    Args:
        depth: Interleave depth (default: 8)

    Returns:
        interleave: Function (symbols) -> interleaved_symbols
        deinterleave: Function (interleaved_symbols) -> symbols
    """
    symbols_per_codeword = 16  # 128 bits / 8 bits per symbol

    def interleave(symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Apply 8-way symbol interleaving.

        Input: symbols from consecutive Hamming codewords
        Output: interleaved symbols

        Per spec Figure 34: symbols are read column-wise from 8×16 matrix.
        """
        symbols = jnp.atleast_2d(symbols)
        n_symbols = symbols.shape[0]

        # Number of complete interleave groups
        group_size = depth * symbols_per_codeword  # 8 × 16 = 128
        n_groups = n_symbols // group_size

        if n_groups == 0:
            return symbols

        # Reshape into groups, then interleave within each group
        groups = symbols[:n_groups * group_size].reshape(n_groups, depth, symbols_per_codeword, -1)

        # Interleave: transpose axes 1 and 2 (codeword index and symbol index)
        # Original: [group, codeword, symbol, pol]
        # After: [group, symbol, codeword, pol] then flatten middle dims
        interleaved = groups.transpose(0, 2, 1, 3).reshape(n_groups * group_size, -1)

        # Append any remaining symbols
        remainder = symbols[n_groups * group_size:]
        if remainder.shape[0] > 0:
            interleaved = jnp.concatenate([interleaved, remainder], axis=0)

        return interleaved

    def deinterleave(symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Reverse 8-way symbol interleaving.
        """
        symbols = jnp.atleast_2d(symbols)
        n_symbols = symbols.shape[0]

        group_size = depth * symbols_per_codeword
        n_groups = n_symbols // group_size

        if n_groups == 0:
            return symbols

        # Reshape and reverse the transpose
        groups = symbols[:n_groups * group_size].reshape(n_groups, symbols_per_codeword, depth, -1)
        deinterleaved = groups.transpose(0, 2, 1, 3).reshape(n_groups * group_size, -1)

        remainder = symbols[n_groups * group_size:]
        if remainder.shape[0] > 0:
            deinterleaved = jnp.concatenate([deinterleaved, remainder], axis=0)

        return deinterleaved

    return interleave, deinterleave


# =============================================================================
# 400ZR TX Chain
# =============================================================================

def TX_chain_400ZR(include_dsp_framing: bool = True):
    """
    400ZR transmitter chain (post-staircase FEC).

    Pipeline:
        bits → Scrambler → Conv Interleaver → Hamming Encode
            → Symbol Map → Symbol Interleave → [DSP Framing] → symbols

    Args:
        include_dsp_framing: If True, include DSP framing (TS, pilots, FAW)

    Returns:
        tx: Function (bits) -> symbols
        config: Chain configuration

    Example:
        tx, cfg = TX_chain_400ZR()
        symbols = tx(payload_bits)
    """
    # Initialize components
    scramble, _ = FrameScrambler()
    ci_interleave, _ = ConvolutionalInterleaver()
    hamming_encode, _ = Hamming128_119()
    sym_map, _ = symbol_mapper_16QAM('400ZR')
    sym_interleave, _ = SymbolInterleaver()

    if include_dsp_framing:
        dsp_frame, _ = DSP_superframe('400ZR')

    def tx(bits: jnp.ndarray) -> jnp.ndarray:
        """
        Transmit chain: bits to symbols.

        Args:
            bits: Input bits (should be multiple of 119 for Hamming blocks)

        Returns:
            symbols: DP-16QAM symbols, shape (N, 2)
        """
        bits = jnp.atleast_1d(bits).astype(jnp.uint8)

        # 1. Scramble
        scrambled = scramble(bits, reset=True)

        # 2. Convolutional interleave (operates on 119-bit blocks)
        interleaved = ci_interleave(scrambled)

        # 3. Hamming encode each 119-bit block to 128 bits
        n_blocks = interleaved.shape[0] // HAMMING_K
        blocks = interleaved[:n_blocks * HAMMING_K].reshape(n_blocks, HAMMING_K)

        def encode_block(block):
            return hamming_encode(block)

        codewords = lax.map(encode_block, blocks)  # (n_blocks, 128)
        encoded_bits = codewords.reshape(-1)

        # 4. Symbol mapping (128 bits -> 16 symbols per codeword)
        symbols = sym_map(encoded_bits)  # (n_symbols, 2)

        # 5. Symbol interleaving (8-way)
        symbols = sym_interleave(symbols)

        # 6. DSP framing (optional)
        if include_dsp_framing:
            symbols = dsp_frame(symbols)

        return symbols

    return tx, CONFIG


def TX_inner_fec_400ZR():
    """
    400ZR inner FEC only (Hamming + interleaver).

    Simplified chain for testing:
        bits → Hamming Encode → symbols

    Returns:
        encode: Function (bits) -> codeword_bits
        decode: Function (codeword_bits) -> bits
    """
    hamming_encode, hamming_decode = Hamming128_119()

    def encode(bits: jnp.ndarray) -> jnp.ndarray:
        """Encode bits through Hamming."""
        bits = jnp.atleast_1d(bits).astype(jnp.uint8)
        n_blocks = bits.shape[0] // HAMMING_K
        blocks = bits[:n_blocks * HAMMING_K].reshape(n_blocks, HAMMING_K)

        codewords = lax.map(hamming_encode, blocks)
        return codewords.reshape(-1)

    def decode(codeword_bits: jnp.ndarray) -> jnp.ndarray:
        """Decode Hamming codewords to bits."""
        codeword_bits = jnp.atleast_1d(codeword_bits).astype(jnp.uint8)
        n_blocks = codeword_bits.shape[0] // HAMMING_N
        codewords = codeword_bits[:n_blocks * HAMMING_N].reshape(n_blocks, HAMMING_N)

        def decode_block(cw):
            msg, _ = hamming_decode(cw)
            return msg

        messages = lax.map(decode_block, codewords)
        return messages.reshape(-1)

    return encode, decode


# =============================================================================
# 400ZR RX Chain
# =============================================================================

def RX_chain_400ZR(include_dsp_framing: bool = True, soft_decision: bool = False):
    """
    400ZR receiver chain (pre-staircase FEC).

    Pipeline:
        symbols → [DSP Deframing] → Symbol Deinterleave → Symbol Demap
            → Hamming Decode → Conv Deinterleave → Descramble → bits

    Args:
        include_dsp_framing: If True, include DSP deframing
        soft_decision: If True, use soft-decision Hamming decoder

    Returns:
        rx: Function (symbols) -> bits
        config: Chain configuration

    Example:
        rx, cfg = RX_chain_400ZR()
        bits = rx(received_symbols)
    """
    # Initialize components
    _, descramble = FrameScrambler()
    _, ci_deinterleave = ConvolutionalInterleaver()

    if soft_decision:
        _, hamming_decode = Hamming128_119_soft()
    else:
        _, hamming_decode = Hamming128_119()

    _, sym_demap = symbol_mapper_16QAM('400ZR')
    _, sym_deinterleave = SymbolInterleaver()

    if include_dsp_framing:
        _, dsp_deframe = DSP_superframe('400ZR')

    def rx(symbols: jnp.ndarray) -> jnp.ndarray:
        """
        Receive chain: symbols to bits.

        Args:
            symbols: Received DP-16QAM symbols, shape (N, 2)

        Returns:
            bits: Decoded bits
        """
        symbols = jnp.atleast_2d(symbols)

        # 1. DSP deframing (optional)
        if include_dsp_framing:
            symbols = dsp_deframe(symbols)

        # 2. Symbol deinterleaving
        symbols = sym_deinterleave(symbols)

        # 3. Symbol demapping
        demapped_bits = sym_demap(symbols)

        # 4. Hamming decode each 128-bit codeword
        n_codewords = demapped_bits.shape[0] // HAMMING_N
        codewords = demapped_bits[:n_codewords * HAMMING_N].reshape(n_codewords, HAMMING_N)

        def decode_block(cw):
            if soft_decision:
                # For soft decision, would need LLRs - use hard bits for now
                msg, _ = hamming_decode(cw.astype(jnp.float32) * 2 - 1)  # Convert to +1/-1
            else:
                msg, _ = hamming_decode(cw)
            return msg

        messages = lax.map(decode_block, codewords)
        decoded = messages.reshape(-1)

        # 5. Convolutional deinterleave
        deinterleaved = ci_deinterleave(decoded)

        # 6. Descramble
        bits = descramble(deinterleaved, reset=True)

        return bits

    return rx, CONFIG


# =============================================================================
# End-to-End Loopback Test
# =============================================================================

def loopback_test(n_bits: int = 119 * 16, include_dsp_framing: bool = False):
    """
    Run end-to-end loopback test of 400ZR chain.

    Args:
        n_bits: Number of bits to test (should be multiple of 119)
        include_dsp_framing: Include DSP framing in test

    Returns:
        success: True if TX->RX roundtrip matches
        stats: Dict with test statistics
    """
    tx, _ = TX_chain_400ZR(include_dsp_framing=include_dsp_framing)
    rx, _ = RX_chain_400ZR(include_dsp_framing=include_dsp_framing)

    # Generate test bits
    n_blocks = n_bits // HAMMING_K
    n_bits_aligned = n_blocks * HAMMING_K
    test_bits = jnp.array([i % 2 for i in range(n_bits_aligned)], dtype=jnp.uint8)

    # TX -> RX
    symbols = tx(test_bits)
    recovered = rx(symbols)

    # Compare (trim to same length)
    min_len = min(test_bits.shape[0], recovered.shape[0])
    match = jnp.array_equal(test_bits[:min_len], recovered[:min_len])

    stats = {
        'input_bits': n_bits_aligned,
        'symbols': symbols.shape[0],
        'recovered_bits': recovered.shape[0],
        'bit_errors': int(jnp.sum(test_bits[:min_len] != recovered[:min_len])),
        'match': bool(match),
    }

    return match, stats

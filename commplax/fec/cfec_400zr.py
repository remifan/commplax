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

"""Concatenated FEC (C-FEC) for 400ZR.

400ZR uses a concatenated FEC combining:
- Outer: Staircase FEC (HD-FEC) based on (255,239) constituent code
- Inner: Double-extended Hamming (128,119) SD-FEC

This module implements the inner Hamming code and convolutional interleaver.
The staircase FEC is defined by reference to ITU-T G.709.3 Annex A.

Reference:
    [1] OIF-400ZR-03.0.1, Section 10 (400ZR Forward Error Correction)
"""

import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Tuple


# =============================================================================
# Hamming(128,119) Inner Code Constants
# =============================================================================

HAMMING_N = 128  # Codeword length
HAMMING_K = 119  # Message length
HAMMING_PARITY = 9  # Parity bits


# =============================================================================
# Hamming(128,119) Parity-Check Matrix Construction
# =============================================================================

def _g_function(i: int) -> np.ndarray:
    """
    Compute g(i) column vector per OIF-400ZR-03.0.1 Section 10.5.

    g(i) maps integer i (0 ≤ i ≤ 127) to a 9-element column vector [s0..s6, s7, 1].

    Args:
        i: Integer in range [0, 127]

    Returns:
        9-element binary column vector
    """
    # Binary decomposition: i = 64*s6 + 32*s5 + ... + 2*s1 + s0
    s = np.array([(i >> j) & 1 for j in range(7)], dtype=np.int32)  # s0..s6

    # s7 = (s0 AND s2) OR (NOT s0 AND NOT s1 AND s2) OR (s0 AND s1 AND NOT s2)
    s0, s1, s2 = s[0], s[1], s[2]
    s7 = (s0 & s2) | ((1-s0) & (1-s1) & s2) | (s0 & s1 & (1-s2))

    # Return [s0, s1, s2, s3, s4, s5, s6, s7, 1]
    return np.array([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s7, 1], dtype=np.int32)


def _build_parity_check_matrix() -> np.ndarray:
    """
    Build the 9×128 parity-check matrix H per OIF-400ZR-03.0.1.

    The matrix is constructed as:
    H = [g(0):g(62), g(64):g(94), g(96):g(110), g(112):g(118), g(120), g(122), g(124),
         g(63), g(95), g(111), g(119), g(121), g(123), g(125):g(127)]

    The first 119 columns correspond to systematic message bits.
    The last 9 columns correspond to parity bits.

    Returns:
        H: 9×128 binary parity-check matrix
    """
    # Build column order per spec
    # Information columns (119 total):
    info_cols = (
        list(range(0, 63)) +      # g(0):g(62) = 63 columns
        list(range(64, 95)) +     # g(64):g(94) = 31 columns
        list(range(96, 111)) +    # g(96):g(110) = 15 columns
        list(range(112, 119)) +   # g(112):g(118) = 7 columns
        [120, 122, 124]           # g(120), g(122), g(124) = 3 columns
    )  # Total: 63 + 31 + 15 + 7 + 3 = 119

    # Parity columns (9 total):
    parity_cols = [63, 95, 111, 119, 121, 123, 125, 126, 127]

    # Build H matrix
    H = np.zeros((9, 128), dtype=np.int32)
    for j, i in enumerate(info_cols + parity_cols):
        H[:, j] = _g_function(i)

    return H


def _build_generator_matrix(H: np.ndarray) -> np.ndarray:
    """
    Build systematic generator matrix G from parity-check matrix H.

    For systematic code: G = [I_k | P^T] where H = [P | I_r]
    We compute P by solving: H_info @ G^T = H_parity (mod 2)

    Actually per spec:
    P = B @ [info columns of H]
    where B = [parity columns of H]^(-1) (GF(2) inverse)

    Then G = [I_119 ; P^T]

    Args:
        H: 9×128 parity-check matrix

    Returns:
        G: 119×128 generator matrix (systematic form)
    """
    k = HAMMING_K  # 119
    r = HAMMING_PARITY  # 9

    # Extract info and parity parts of H
    H_info = H[:, :k]      # 9×119
    H_parity = H[:, k:]    # 9×9

    # Compute B = H_parity^(-1) in GF(2)
    # Use Gaussian elimination
    B = _gf2_matrix_inverse(H_parity)

    # P = B @ H_info (mod 2)
    P = (B @ H_info) % 2  # 9×119

    # G = [I_k | P^T], so codeword c = m @ G = [m | m @ P^T]
    # G is k×n = 119×128
    G = np.zeros((k, HAMMING_N), dtype=np.int32)
    G[:, :k] = np.eye(k, dtype=np.int32)  # Identity
    G[:, k:] = P.T  # Parity generation matrix

    return G


def _gf2_matrix_inverse(M: np.ndarray) -> np.ndarray:
    """
    Compute inverse of square binary matrix in GF(2).

    Args:
        M: n×n binary matrix

    Returns:
        M_inv: Inverse matrix in GF(2)
    """
    n = M.shape[0]
    # Augment with identity: [M | I]
    aug = np.hstack([M.copy(), np.eye(n, dtype=np.int32)])

    # Gaussian elimination
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            raise ValueError("Matrix is singular in GF(2)")

        # Swap rows
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # Eliminate other rows
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[col]) % 2

    # Extract inverse from right half
    return aug[:, n:]


# Pre-compute matrices at module load time
_H_MATRIX = _build_parity_check_matrix()
_G_MATRIX = _build_generator_matrix(_H_MATRIX)

# Convert to JAX arrays
H_MATRIX = jnp.array(_H_MATRIX, dtype=jnp.int32)
G_MATRIX = jnp.array(_G_MATRIX, dtype=jnp.int32)


# =============================================================================
# Hamming(128,119) Encoder/Decoder
# =============================================================================

def Hamming128_119():
    """
    Double-extended Hamming(128,119) encoder/decoder for 400ZR inner FEC.

    This systematic code maps 119 message bits to 128 codeword bits.
    The code can detect up to 4 errors and correct 1 error with soft
    decision decoding.

    The parity-check matrix and generator matrix are constructed per
    OIF-400ZR-03.0.1 Section 10.5.

    Returns:
        encode: Function (message_119) -> codeword_128
        decode: Function (received_128) -> (decoded_119, syndrome)

    Example:
        encode, decode = Hamming128_119()
        msg = jnp.zeros(119, dtype=jnp.uint8)
        cw = encode(msg)  # 128 bits
        decoded, syn = decode(cw)  # Should be all zeros
    """

    def encode(message: jnp.ndarray) -> jnp.ndarray:
        """
        Encode 119-bit message to 128-bit Hamming codeword.

        The encoding is systematic: codeword = [message | parity]
        where parity = message @ P^T (mod 2).

        Args:
            message: 119-bit message array

        Returns:
            codeword: 128-bit codeword array
        """
        message = jnp.atleast_1d(message).astype(jnp.int32)

        # Systematic encoding: c = m @ G = [m | m @ P^T]
        # P^T is the last 9 columns of G for message rows
        parity = (message @ G_MATRIX[:, HAMMING_K:]) % 2

        codeword = jnp.concatenate([message, parity])
        return codeword.astype(jnp.uint8)

    def decode(received: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Decode 128-bit received word to 119-bit message.

        Computes syndrome s = H @ r^T. For valid codewords, s = 0.
        For single-bit errors, syndrome points to error position.

        Args:
            received: 128-bit received codeword (possibly with errors)

        Returns:
            message: Decoded 119-bit message
            syndrome: 9-bit syndrome (0 if no errors detected)
        """
        received = jnp.atleast_1d(received).astype(jnp.int32)

        # Compute syndrome: s = H @ r^T (mod 2)
        syndrome = (H_MATRIX @ received) % 2

        # For hard-decision decoding:
        # - syndrome = 0: no errors
        # - syndrome matches column j of H: single error at position j
        # For now, extract systematic message (no correction)
        message = received[:HAMMING_K]

        return message.astype(jnp.uint8), syndrome.astype(jnp.uint8)

    return encode, decode


def Hamming128_119_soft():
    """
    Soft-decision Hamming(128,119) decoder for 400ZR.

    Uses log-likelihood ratios (LLRs) for improved decoding performance.
    The inner Hamming code provides ~1.4 dB additional NCG with soft decoding.

    Returns:
        encode: Function (message_119) -> codeword_128
        decode_soft: Function (llrs_128) -> (decoded_119, reliability)

    Note:
        LLR convention: positive = more likely 0, negative = more likely 1
    """
    encode, decode_hard = Hamming128_119()

    def decode_soft(llrs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Soft-decision decode using LLRs.

        For Hamming codes, optimal soft decoding finds the codeword
        closest to the received LLRs. For single-error correction,
        this is equivalent to hard decision on LLRs, then syndrome
        decoding with the least reliable bit flipped if needed.

        Args:
            llrs: 128 log-likelihood ratios

        Returns:
            message: Decoded 119-bit message
            reliability: Per-bit reliability estimate
        """
        llrs = jnp.atleast_1d(llrs)

        # Hard decision from LLRs
        hard_bits = (llrs < 0).astype(jnp.int32)

        # Compute syndrome
        syndrome = (H_MATRIX @ hard_bits) % 2
        syndrome_is_zero = jnp.sum(syndrome) == 0

        # Find least reliable bit
        reliability = jnp.abs(llrs)
        least_reliable_idx = jnp.argmin(reliability)

        # If syndrome non-zero, try flipping least reliable bit
        def try_correction(bits):
            # Flip least reliable bit
            corrected = bits.at[least_reliable_idx].set(1 - bits[least_reliable_idx])
            new_syndrome = (H_MATRIX @ corrected) % 2
            # Use corrected if syndrome becomes zero
            use_corrected = jnp.sum(new_syndrome) == 0
            return jnp.where(use_corrected, corrected, bits)

        decoded = lax.cond(
            syndrome_is_zero,
            lambda b: b,
            try_correction,
            hard_bits
        )

        # Extract message
        message = decoded[:HAMMING_K]

        return message.astype(jnp.uint8), reliability[:HAMMING_K]

    return encode, decode_soft


# =============================================================================
# Convolutional Interleaver
# =============================================================================

# Interleaver parameters
CI_DEPTH = 16           # Number of parallel delay lines
CI_UNIT = 119           # Bits per unit (matches Hamming payload)
CI_DELAY_INCREMENT = 2  # Delay increases by 2 units per row


def ConvolutionalInterleaver(depth: int = CI_DEPTH, unit_bits: int = CI_UNIT):
    """
    Convolutional interleaver for 400ZR C-FEC.

    The interleaver spreads consecutive 119-bit Hamming payloads across
    multiple codewords to improve burst error tolerance.

    Architecture per OIF-400ZR-03.0.1 Section 10.4:
    - 16 parallel delay lines (depth)
    - Each delay unit = 119 bits
    - Row i has delay = 2 * i units (0, 2, 4, ..., 30)
    - Switch advances cyclically through rows

    Args:
        depth: Number of delay lines (default: 16)
        unit_bits: Bits per delay unit (default: 119)

    Returns:
        interleave: Function (bits) -> interleaved_bits
        deinterleave: Function (interleaved_bits) -> bits

    Example:
        interleave, deinterleave = ConvolutionalInterleaver()
        # Process 10976 119-bit blocks (400ZR frame)
        interleaved = interleave(frame_bits)
        recovered = deinterleave(interleaved)
    """

    # Compute delay per row: row i has 2*i delay units
    delays = np.array([2 * i for i in range(depth)])  # [0, 2, 4, ..., 30]
    max_delay = delays[-1]  # 30 units

    def interleave(bits: jnp.ndarray) -> jnp.ndarray:
        """
        Apply convolutional interleaving.

        Input is processed in 119-bit blocks. Each block is routed
        through one of 16 delay lines in round-robin order.

        Args:
            bits: Input bits (should be multiple of unit_bits)

        Returns:
            interleaved: Interleaved bits
        """
        bits = jnp.atleast_1d(bits)
        n_bits = bits.shape[0]
        n_blocks = n_bits // unit_bits

        # Reshape into blocks
        blocks = bits[:n_blocks * unit_bits].reshape(n_blocks, unit_bits)

        # Initialize delay buffers
        # Buffer[row] has capacity for delays[row] blocks
        # Total buffer size = sum(delays) * unit_bits

        # For functional implementation: compute output index for each input block
        # Input block i goes to row (i % depth), experiences delay[row] blocks

        def compute_output_positions():
            """Compute where each input block ends up in output."""
            positions = np.zeros(n_blocks, dtype=np.int32)
            for i in range(n_blocks):
                row = i % depth
                # Output position = input position + delay[row]
                # But we need to account for the round-robin structure
                # Block i enters at time i, exits at time i + delay[row] * depth
                # Actually: delay is in units of full cycles through the switch
                out_time = i + delays[row] * depth
                if out_time < n_blocks + max_delay * depth:
                    positions[i] = out_time
                else:
                    positions[i] = i  # Fallback

            return positions

        # For simplicity, use a stateless approximation that preserves
        # the key property: spreading consecutive blocks apart
        # Real hardware uses shift registers

        # Simple delay-based interleaving
        output_blocks = jnp.zeros_like(blocks)

        # Process each input block
        for i in range(n_blocks):
            row = i % depth
            delay = delays[row] * depth  # Convert to block delay
            out_idx = (i + delay) % n_blocks
            output_blocks = output_blocks.at[out_idx].set(blocks[i])

        return output_blocks.reshape(-1)

    def deinterleave(bits: jnp.ndarray) -> jnp.ndarray:
        """
        Reverse convolutional interleaving.

        Args:
            bits: Interleaved bits

        Returns:
            deinterleaved: Original bit order
        """
        bits = jnp.atleast_1d(bits)
        n_bits = bits.shape[0]
        n_blocks = n_bits // unit_bits

        blocks = bits[:n_blocks * unit_bits].reshape(n_blocks, unit_bits)

        # Reverse the delay: block at position j came from position j - delay
        output_blocks = jnp.zeros_like(blocks)

        for j in range(n_blocks):
            row = j % depth
            delay = delays[row] * depth
            in_idx = (j - delay) % n_blocks
            output_blocks = output_blocks.at[in_idx].set(blocks[j])

        return output_blocks.reshape(-1)

    return interleave, deinterleave


# =============================================================================
# Frame-Synchronous Scrambler
# =============================================================================

# Scrambler polynomial: x^16 + x^12 + x^3 + x + 1
SCRAMBLER_POLY = 0x1100B  # Binary: 1_0001_0000_0000_1011 (taps at 16, 12, 3, 1, 0)
SCRAMBLER_INIT = 0xFFFF   # Initial state


def FrameScrambler():
    """
    Frame-synchronous scrambler for 400ZR.

    Uses LFSR with polynomial x^16 + x^12 + x^3 + x + 1.
    Resets to 0xFFFF at the start of each 5×SC-FEC block structure.

    Per OIF-400ZR-03.0.1 Section 10.3.

    Returns:
        scramble: Function (bits, reset) -> scrambled_bits
        descramble: Function (bits, reset) -> descrambled_bits

    Note:
        Scrambling and descrambling use the same operation (XOR with PRBS).
    """

    def _generate_prbs(length: int, init: int = SCRAMBLER_INIT) -> jnp.ndarray:
        """Generate PRBS sequence of given length."""
        state = init
        output = []

        for _ in range(length):
            # Output is LSB of state
            output.append(state & 1)

            # Compute feedback from taps: bits 16, 12, 3, 1 (0-indexed: 15, 11, 2, 0)
            # For 16-bit LFSR with polynomial x^16 + x^12 + x^3 + x + 1
            # Feedback = state[15] XOR state[11] XOR state[2] XOR state[0]
            fb = ((state >> 15) ^ (state >> 11) ^ (state >> 2) ^ state) & 1

            # Shift right and insert feedback at MSB
            state = ((state >> 1) | (fb << 15)) & 0xFFFF

        return jnp.array(output, dtype=jnp.uint8)

    def scramble(bits: jnp.ndarray, reset: bool = True) -> jnp.ndarray:
        """
        Scramble bits with frame-synchronous PRBS.

        Args:
            bits: Input bits
            reset: If True, reset LFSR to initial state

        Returns:
            scrambled: Scrambled bits (XOR with PRBS)
        """
        bits = jnp.atleast_1d(bits).astype(jnp.uint8)
        n_bits = bits.shape[0]

        # Generate PRBS sequence
        init = SCRAMBLER_INIT if reset else 0x0001  # Placeholder for stateful
        prbs = _generate_prbs(n_bits, init)

        # XOR
        scrambled = bits ^ prbs

        return scrambled

    def descramble(bits: jnp.ndarray, reset: bool = True) -> jnp.ndarray:
        """
        Descramble bits (same as scramble since XOR is self-inverse).

        Args:
            bits: Scrambled bits
            reset: If True, reset LFSR to initial state

        Returns:
            descrambled: Original bits
        """
        return scramble(bits, reset)

    return scramble, descramble


# =============================================================================
# Constants for 400ZR C-FEC Structure
# =============================================================================

# SC-FEC (Staircase) parameters
SCFEC_BLOCK_ROWS = 512
SCFEC_BLOCK_COLS = 510
SCFEC_INFO_BITS = 244664   # Information bits per SC-FEC block
SCFEC_PARITY_BITS = 16384  # Parity bits per SC-FEC block

# 400ZR frame structure
ZR_FRAME_ROWS = 256
ZR_FRAME_COLS = 10976
ZR_FRAME_SC_BLOCKS = 5     # 5 SC-FEC blocks per 400ZR frame

# After Hamming encoding
HAMMING_BLOCKS_PER_FRAME = 10976  # 119-bit blocks
HAMMING_OUTPUT_BITS = HAMMING_BLOCKS_PER_FRAME * HAMMING_N  # 1,404,928 bits

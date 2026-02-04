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

"""Forward Error Correction (FEC) kernels.

This module provides FEC encoder/decoder kernels for coherent optical systems.

400ZR C-FEC (OIF-400ZR-03.0):
    - Hamming128_119: Double-extended Hamming inner code (SD-FEC)
    - Hamming128_119_soft: Soft-decision Hamming decoder
    - ConvolutionalInterleaver: Depth-16 convolutional interleaver
    - FrameScrambler: Frame-synchronous scrambler

1600ZR+ OFEC (OIF 1600ZR+ IA):
    - OFEC: Open FEC encoder/decoder (8 parallel BCH encoders + 4 interleavers)
    - OFEC_interleaver: Standalone interleaver kernel
    - BCH256_239: BCH constituent code encoder/decoder

Example:
    # 400ZR inner FEC
    from commplax.fec import Hamming128_119, ConvolutionalInterleaver
    encode, decode = Hamming128_119()
    interleave, deinterleave = ConvolutionalInterleaver()

    # Encode 119-bit message
    codeword = encode(message)  # -> 128 bits
    interleaved = interleave(codewords)

    # 1600ZR+ OFEC
    from commplax.fec import OFEC
    encode, decode = OFEC()
"""

# 1600ZR+ OFEC
from .ofec import (
    OFEC,
    OFEC_interleaver,
    BCH256_239,
    BCH_GENERATOR_POLY,
    BCH_N,
    BCH_K,
    OFEC_BLOCK_INPUT_BITS,
    OFEC_ENCODER_INPUT_BITS,
    OFEC_ENCODER_OUTPUT_BITS,
)

# 400ZR C-FEC
from .cfec_400zr import (
    # Hamming inner code
    Hamming128_119,
    Hamming128_119_soft,
    HAMMING_N,
    HAMMING_K,
    HAMMING_PARITY,
    H_MATRIX,
    G_MATRIX,
    # Convolutional interleaver
    ConvolutionalInterleaver,
    CI_DEPTH,
    CI_UNIT,
    # Frame scrambler
    FrameScrambler,
    SCRAMBLER_POLY,
    SCRAMBLER_INIT,
    # SC-FEC constants
    SCFEC_BLOCK_ROWS,
    SCFEC_BLOCK_COLS,
    SCFEC_INFO_BITS,
    SCFEC_PARITY_BITS,
    ZR_FRAME_ROWS,
    ZR_FRAME_COLS,
    ZR_FRAME_SC_BLOCKS,
    HAMMING_BLOCKS_PER_FRAME,
    HAMMING_OUTPUT_BITS,
)

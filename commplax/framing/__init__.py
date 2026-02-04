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

"""Frame structure kernels for coherent optical systems.

This module provides framing/deframing kernels for OIF ZR systems:
    - 400ZR (OIF-400ZR-03.0)
    - 800ZR (OIF-800ZR-01.0)
    - 1600ZR+ (OIF 1600ZR+ IA)

Available kernels:
    - dsp_frame: DSP sub-frame, super-frame, symbol mapping, pilot generation
    - flexo: FlexO/100G ZR frame structure (placeholder)

Example:
    from commplax.framing import DSP_subframe, DSP_superframe, get_config

    # 400ZR DSP framing
    frame_fn, deframe_fn = DSP_subframe('400ZR')
    subframe = frame_fn(data_symbols, subframe_index=0)

    # 1600ZR+ DSP framing
    frame_fn, deframe_fn = DSP_superframe('1600ZR+')
    superframe = frame_fn(data_symbols)
"""

from .flexo import FlexO_frame, ZR_instance
from .dsp_frame import (
    # Configuration
    DSPFrameConfig,
    CONFIG_400ZR,
    CONFIG_800ZR,
    CONFIG_1600ZRP,
    CONFIGS,
    get_config,
    # Symbol mapping
    symbol_mapper_16QAM,
    GRAY_MAP,
    # Training and pilots
    get_training_sequence,
    get_faw,
    generate_pilot_sequence,
    # DSP framing kernels
    DSP_subframe,
    DSP_superframe,
    # Interleaver (1600ZR+ specific)
    interleaver_merge,
    subcarrier_bit_distribute,
    # 400ZR constants
    SUBFRAME_SYMBOLS_400ZR,
    SUPERFRAME_SUBFRAMES_400ZR,
    SUPERFRAME_SYMBOLS_400ZR,
    PILOT_INTERVAL_400ZR,
    TS_SYMBOLS_400ZR,
    # 1600ZR+ constants (backward compat)
    SUBFRAME_SYMBOLS,
    SUPERFRAME_SUBFRAMES,
    SUPERFRAME_SYMBOLS,
    PAYLOAD_SYMBOLS,
    PILOT_INTERVAL,
    TS_SYMBOLS,
    FAW_SYMBOLS,
)

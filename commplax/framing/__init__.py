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

"""Frame structure kernels for coherent optical systems.

This module provides framing/deframing kernels for 1600ZR/ZR+ systems.

Available kernels:
    - flexo: FlexO/100G ZR frame structure
    - dsp_frame: DSP sub-frame and super-frame structure
"""

from .flexo import FlexO_frame, ZR_instance
from .dsp_frame import DSP_subframe, DSP_superframe

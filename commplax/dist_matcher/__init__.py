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

"""Distribution Matching for Probabilistic Constellation Shaping.

This package provides distribution matchers for optical communication systems:

- CCDM: Constant Composition Distribution Matching (arithmetic coding based)
- HierDM/PCS: Hierarchical LUT-based Distribution Matching (for ZR+)

Example:
    >>> from commplax.dist_matcher import CCDM, CCDM_helper
    >>> from commplax.dist_matcher import PCS, HierDM

References:
    [1] Schulte & Böcherer, "Constant composition distribution matching", IEEE TIT 2015
    [2] OIF 1600ZR+ Implementation Agreement, Section 6.3
"""

from commplax.dist_matcher.ccdm import (
    CCDM,
    CCDM_helper,
    idquant,
    quant_to_ntype,
    find_n_by_maxbitnum,
    n_choose_k_log2,
    n_choose_ks_log2,
)

from commplax.dist_matcher.hierdm import (
    HierDM,
    PCS,
    LUTConfig,
    HIERARCHY_1600ZRP,
    HIERARCHY_1200ZRP,
)

__all__ = [
    # CCDM
    'CCDM',
    'CCDM_helper',
    'idquant',
    'quant_to_ntype',
    'find_n_by_maxbitnum',
    'n_choose_k_log2',
    'n_choose_ks_log2',
    # Hierarchical DM / PCS
    'HierDM',
    'PCS',
    'LUTConfig',
    'HIERARCHY_1600ZRP',
    'HIERARCHY_1200ZRP',
]

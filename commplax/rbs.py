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


import jax
from jax import numpy as jnp, lax
import numpy as np
from typing import Literal


def prbs_n(
    order: Literal[7, 9, 11, 15, 20, 23, 31],
    size: int = None,
    seed: int = None,
    return_seed: bool = False,
):
    """Pseudorandom binary sequence generator

    Args:
    order: must be one of the following values (7, 9, 11, 15, 20, 23, 31)
    seed: int, can't be 0 or a multiple of 2**order
    """
    taps = {
        7 : [6 , 5 ],
        9 : [8 , 4 ],
        11: [10, 8 ],
        15: [14, 13],
        20: [19, 2 ],
        23: [22, 17],
        31: [30, 27],
    }
    seed = seed % (2**order) if seed is not None else (1 << order) - 1
    size = 2**order - 1 if size is None else size

    lfsr = jnp.array(seed, dtype=jnp.uint32)  # initial state of the LFSR
    tap1, tap2 = taps[order]

    def step(lfsr, _):
        y = lfsr & 1
        new = ((lfsr >> tap1) ^ (lfsr >> tap2)) & 1
        lfsr = ((lfsr << 1) | new) & (1 << order) - 1
        return lfsr, y

    lfsr, output = lax.scan(step, lfsr, jnp.empty(size))

    if not return_seed:
        return output
    return output, lfsr



def anuqrng_bit(L):
    ''' https://github.com/lmacken/quantumrandom
    import quantumrandom
    '''
    L    = int(L)
    N    = 0
    bits = []

    while N < L:
        b = np.unpackbits(np.frombuffer(quantumrandom.binary(), dtype=np.uint8))
        N += len(b)
        bits.append(b)

    bits = jnp.concatenate(bits)[:L]

    return bits


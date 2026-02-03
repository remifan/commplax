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

"""Module composition utilities for equinox-based DSP modules.

This module provides utilities for:
- Scanning modules over sequences via lax.scan
- Composable step functions with all-reduce patterns for ensembles

For creating ensembles, use ``eqx.filter_vmap`` directly.
See: https://docs.kidger.site/equinox/tricks/#ensembling
"""

import dataclasses as dc
import jax
from jax import lax, numpy as jnp
from typing import Callable, Optional


def scan_with(step: Optional[Callable] = None, jit_backend: Optional[str] = None):
    """Create a scanner for equinox modules.

    Factory that returns a JIT-compiled function for scanning a module
    over a sequence of inputs using lax.scan.

    Args:
        step: Custom step function (state, x) -> (state, y).
            If None, uses module's __call__: m(x) -> (m, y).
        jit_backend: JIT backend option:
            - 'cpu': Force CPU backend (often faster for symbol-wise ops)
            - 'gpu'/'tpu': Force specific backend
            - None: Use JAX default
            - False: Disable JIT

    Returns:
        A function (module, xs) -> (updated_module, ys).

    Example:
        # Simple: scan module's __call__
        scanner = scan_with()
        mimo_updated, outputs = scanner(mimo, inputs)

        # Force CPU backend for symbol-wise processing
        scanner = scan_with(jit_backend='cpu')
        mimo_updated, outputs = scanner(mimo, inputs)

        # Custom step with ensemble processing
        def foe_step(foe, x):
            foe, aux = eqx.filter_vmap(eq.FOE.update)(foe, x)
            fo_avg = jnp.mean(foe.fo) * jnp.ones_like(foe.fo)
            foe = dc.replace(foe, fo=fo_avg)
            foe, y = eqx.filter_vmap(eq.FOE.apply)(foe, x)
            return foe, (y, aux)

        scanner = scan_with(foe_step, jit_backend='cpu')
        foe_updated, (y, aux) = scanner(foe, x_blocks)
    """
    def scan_fn(m, x):
        # step is not hashable, dont't do
        # f = (lambda m, x: m(x)) if step is None else step
        # instead do
        f = lambda m, x: m(x) if step is None else step(m, x)
        return lax.scan(f, m, x)

    if jit_backend is False:
        return scan_fn
    elif jit_backend is not None:
        return jax.jit(scan_fn, backend=jit_backend)
    else:
        return jax.jit(scan_fn)


# =============================================================================
# Composable ensemble step primitives
# =============================================================================

def allreduce(field: str, op: Callable = jnp.mean):
    """All-reduce: reduce a field across ensemble and broadcast back.

    This is the "cross-compute" step in the update → reduce → apply pattern.

    Args:
        field: Name of the field to reduce (e.g., 'fo' for frequency offset).
        op: Reduction operation (default: jnp.mean). Common choices:
            - jnp.mean: average across ensemble
            - jnp.median: robust to outliers
            - lambda x: x[0]: use first element only

    Returns:
        A function (ensemble, x) -> (ensemble_with_reduced_field, x).

    Example:
        reduce_fo = allreduce('fo', jnp.mean)
        foe, x = reduce_fo(foe, x)  # average FO across polarizations
    """
    def fn(ensemble, x_aux):
        value = getattr(ensemble, field)
        reduced = op(value) * jnp.ones_like(value)
        ensemble = dc.replace(ensemble, **{field: reduced})
        return ensemble, x_aux
    return fn


def pipe(*fns):
    """Compose multiple step functions into a single pipeline.

    Each function should have signature (state, x) -> (state, x).
    Functions are applied left-to-right.

    Args:
        *fns: Step functions to compose.

    Returns:
        A composed function (state, x) -> (state, x).

    Example:
        # Compose multiple transformations
        step = pipe(
            lambda state, x: (state, preprocess(x)),
            lambda state, x: model(state, x),
            lambda state, x: (state, postprocess(x)),
        )

        # Use with lax.scan
        final_state, outputs = lax.scan(step, init_state, inputs)
    """
    def piped(state, x):
        for f in fns:
            state, x = f(state, x)
        return state, x
    return piped

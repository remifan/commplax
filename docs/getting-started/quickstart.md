# Quick Start

## Basic Usage

Commplax modules are [Equinox](https://github.com/patrick-kidger/equinox) PyTrees, making them compatible with all JAX transforms (jit, vmap, grad, etc.).

### Create a Module

```python
from commplax import equalizer as eq, adaptive_kernel as ak

# Create a 15-tap 2x2 MIMO equalizer with RLS-CMA
mimo = eq.MIMOCell(
    num_taps=15,
    kernel=ak.rls_cma(),
    dims=2,
)
```

### Run with scan_with

```python
from commplax import module as mod

# Run over signal (JIT-compiled by default)
mimo_updated, output = mod.scan_with()(mimo, signal)
```

### Create Ensembles with filter_vmap

```python
import equinox as eqx
from jax import numpy as jnp

# Create 2 independent CPR modules (one per polarization)
@eqx.filter_vmap
def make_cpr(_):
    return eq.CPR(kernel=ak.cpr_partition_pll(mu=0.01))

cpr = make_cpr(jnp.arange(2))
```

## Next Steps

- See [Examples](../examples/index.md) for complete working notebooks
- Check the [API Reference](../api.md) for detailed documentation

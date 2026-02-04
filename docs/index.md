# Commplax

Functional Optical PHY in JAX.

!!! warning "Under Development"
    Commplax is under heavy development. APIs may change and things can break.

## Overview

Since commplax was first released for the paper [gdbp_study](https://github.com/remifan/gdbp_study), the JAX ecosystem has undergone rapid changes. This version modernizes commplax with better practices around [Equinox](https://github.com/patrick-kidger/equinox) and [JAX](https://github.com/google/jax), informed by industrial perspectives.

## Quick Start

Modules are [Equinox](https://github.com/patrick-kidger/equinox) PyTrees — compatible with all JAX transforms.

```python
from commplax import equalizer as eq, adaptive_kernel as ak, module as mod

# Create module
mimo = eq.MIMOCell(15, kernel=ak.rls_cma(), dims=2)

# Run with scan (JIT-compiled)
mimo, out = mod.scan_with()(mimo, signal)
```

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Examples](examples/index.md)

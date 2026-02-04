# Commplax: JAX-based toolbox for optical communications

> **Note:** Commplax is under heavy development. APIs may change and things can break.

Since commplax was first released for the paper [gdbp_study](https://github.com/remifan/gdbp_study), the JAX ecosystem has undergone rapid changes. This version modernizes commplax with better practices around [Equinox](https://github.com/patrick-kidger/equinox) and [JAX](https://github.com/google/jax).

## Installation

**Requirements:** Python >= 3.11 (3.11 tested)

Editable install is recommended for easier testing and debugging.

```bash
git clone https://github.com/remifan/commplax.git
cd commplax
```

**CPU only** (Windows/Linux/Mac):
```bash
pip install -e "."
```

**CUDA 12** (Linux only, Windows not supported; WSL2 may work but untested):
```bash
pip install -e ".[cuda12]"
```

Verify GPU detection:
```python
import jax; print(jax.devices())
```

> **Note:** CUDA 13 with JAX >= 0.7 is untested. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more details.

## Highlights

Modules are [Equinox](https://github.com/patrick-kidger/equinox) PyTrees — compatible with all JAX transforms.

```python
from commplax import equalizer as eq, adaptive_filter as af, module as mod

# Create module
mimo = eq.MIMOCell(15, af=af.rls_cma(), dims=2)

# Run with scan (JIT-compiled)
mimo, out = mod.scan_with()(mimo, signal)
```

## Examples

Interactive [marimo](https://marimo.io) notebooks — see [examples/](examples/) or [documentation](https://remifan.github.io/commplax/examples/).

## Reference implementations (with sample mode)
PMD layer:
- [x] M/K-spaced NxN MIMO
- [x] Adaptive equalizers(CMA/LMS/RLS/Kalman...)
- [x] Polyphase resampler
- [x] Detector (e.g., Viterbi)
- [x] Timing syncrhonizer
- [x] Distribution matcher (e.g. CCDM)
- [ ] Probabilistic shaping
- [ ] Standards-compliant FEC (e.g., CFEC, oFEC)

Optical channel:
- [ ] fiber/WSS/ROSA/TOSA/...


## Acknowledgement
- [Google/JAX](https://github.com/google/jax)
- [patrick-kidger/equinox](https://github.com/patrick-kidger/equinox)
- [Alan Pak Tao Lau](https://www.alanptlau.org/)
- [Chao Lu](http://www.eie.polyu.edu.hk/~enluchao/)


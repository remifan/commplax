# Commplax: Functional Optical PHY in JAX

> **Note:** Commplax is under heavy development. APIs may change and things can break.

Since commplax was first released for the paper [gdbp_study](https://github.com/remifan/gdbp_study), the JAX ecosystem has undergone rapid changes. This version modernizes commplax with better practices around [Equinox](https://github.com/patrick-kidger/equinox) and [JAX](https://github.com/google/jax), informed by industrial perspectives.

## Installation

**Requirements:** Python >= 3.11

```bash
git clone https://github.com/remifan/commplax.git
cd commplax
pip install -e ".[plot]"          # CPU with visualization
pip install -e ".[plot,cuda12]"   # CUDA 12 with visualization
```

Optional extras: `plot` (visualization), `cuda12` (GPU), `dev` (testing/linting), `docs` (documentation)

> **Note:** See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for GPU setup details.

## Quick Start

Modules are [Equinox](https://github.com/patrick-kidger/equinox) PyTrees — compatible with all JAX transforms.

```python
from commplax import equalizer as eq, adaptive_kernel as ak, module as mod

# Create module
mimo = eq.MIMOCell(15, kernel=ak.rls_cma(), dims=2)

# Run with scan (JIT-compiled)
mimo, out = mod.scan_with()(mimo, signal)
```

## Examples

Interactive [marimo](https://marimo.io) notebooks — see [examples/](examples/) or [documentation](https://remifan.github.io/commplax/examples/).

## Checklist

**Coherent DSP:**
- [x] NxN MIMO equalizer (M/K-spaced butterfly)
- [x] Adaptive kernels (CMA, LMS, RLS, Kalman)
- [x] Carrier phase recovery, frequency offset estimation
- [x] Polyphase resampler, symbol timing recovery
- [x] Viterbi detector
- [x] Sub-carrier mux/demux

**Shaping & Coding:**
- [x] Distribution matcher (CCDM)
- [ ] Probabilistic shaping (PCS)

**FEC:**
- [x] 400ZR inner FEC: Hamming(128,119), conv interleaver, scrambler
- [ ] 400ZR outer FEC: Staircase decoder
- [x] 800ZR/1600ZR+ OFEC: BCH(256,239), interleaver
- [ ] 800ZR/1600ZR+ OFEC: Full staircase/zipper decoder

**Framing:**
- [x] DSP sub-frame / super-frame (400ZR, 800ZR, 1600ZR+)
- [x] DP-16QAM symbol mapping (Gray code)
- [ ] FlexO / 100G ZR instances
- [ ] GMP client mapping

**Channel Models:**
- [x] Fiber propagation (Manakov SSF)
- [x] Optical modulator, laser source
- [ ] Other analogs


## Acknowledgement
- [Google/JAX](https://github.com/google/jax)
- [patrick-kidger/equinox](https://github.com/patrick-kidger/equinox)
- [Alan Pak Tao Lau](https://www.alanptlau.org/)
- [Chao Lu](http://www.eie.polyu.edu.hk/~enluchao/)


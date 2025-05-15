# Commplax: JAX-based toolbox for optical communications

Commplax is resurrected!

Thinking in JAX and JAX your codes.

## new features

### Model as Pytree => seamless integration with JAX
```Python
cpr = eq.CPR()           # create a single dimension carrier-phase-recovery module
cpr = jax.vmap(cpr)      # vmap CPR to be N-dimension (e.g., 2 for dual polarizations)
cpr, output = cpr(input) # input.shape=(2,N) => output.shape=(2,N)
```

### Sample mode => Easy composition
```Python
# An example of 15-tap 3/4-spaced 8x8 MIMO with JIT
mimo = eq.MIMOCell(15, af=cma(lr=5e-5), dims=8, up=3, down=4)
mimo_updated, output = jax.jit(scan)(mimo, input)
```
with sample mode, composition of feedback loop is easy
``` Python
mimo = eq.MIMOCell(15, af=dd_lms, dims=2)
cpr = eq.CPR(dims=2, mode='feedback')

def cpr_mimo_loop(state, x):
    cpr, mimo = state
    y = cpr.apply(x)         # remove carrier phase
    mimo, y = mimo(y)        # apply MIMO
    cpr, y = cpr.update(y)   # update carrier phase tracker
    state = cpr, mimo
    return state, y

(cpr_updated, mimo_updated), output = scan(cpr_mimo_loop, (cpr, mimo), input)
```


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


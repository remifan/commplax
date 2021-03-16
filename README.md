# Commplax: DSP in JAX
Commplax is a Python DSP library written in [JAX](https://github.com/google/jax), which is made by Google for high-performance machine learning research. Thanks to JAX's friendly API (most are Numpy's), efficient Autograd function and hardware acceleration (e.g. JIT in CPU and GPU), Commplax is/can:

- shipped with accelerated well-tested DSP algorithms and core operations
- optimize computationally complex DSP (e.g. Diginal Back Propogation) which is tranditionally inconvenient to do
- optimize DSP with deep learning models written in JAX's derivitives (e.g. [TRAX](https://github.com/google/trax))
- designed carefully to maximize the readlibity and usablity of the codebase
- flawlessly deply to cloud runtime (e.g. Colab, Binder) to share and colabrate

Commplax is designed for researchers in (optical) communication community and machine learning community.
- Tranditional physical layer DSP experts can reply on Commplax to accerate their algorithms, optimize the most complicated parts, and further learn how deep learn works from bottom (autograd, optimizers, backprop, ...) to top (all kinds of layers, structures,....).
- ML researchers can play with Commplax to see the domain specfic parts is DSP (e.g. non-stationary random distortions, fiber non-liearties) and include Commplax's DSP operation as one of their toolbox to improve training capability


## Quickstart
The best to understand how Commplax works is through jupyter's notebook demo, here are some examples
- [Hello world](https://github.com/remifan/commplax/examples/hello_world.ipynb) - demodulate DP-16QAM 815km SSMF signal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/remifan/commplax/blob/master/examples/hello_world.ipynb)
- (In progress) First glance of optimzation - optimize Digital Back Propogation (namely DNN-DBP or LDBP) while adapting DSP
- (In progress) Play with DNN - integrate TRAX

## Installation


## Where to get help


# Commplax: DSP in JAX
Commplax is a modern Python DSP library written in [JAX](https://github.com/google/jax), which is made by Google for high-performance machine learning research. Thanks to JAX's friendly API (most are [Numpy](https://numpy.org/)'s), efficient Autograd function and hardware acceleration (e.g. JIT in CPU and GPU), Commplax is/can:

- deal with Complex number well, thanks to JAX's native Complex number support
- shipped with accelerated well-tested DSP algorithms and core operations
- optimize computationally complex DSP (e.g. Diginal Back Propogation) which is tranditionally inconvenient to do
- optimize DSP with deep learning models written in JAX's derivitives (e.g. [TRAX](https://github.com/google/trax))
- designed carefully to maximize the readlibity and usability of the codebase
- flawlessly deploy to cloud runtime (e.g. [Colab](https://colab.research.google.com/), [Binder](https://mybinder.org/)) to share and colabrate

Commplax is designed for researchers in (optical) communication community and machine learning community, and hopefully may help to ease the collaboration between 2 worlds.
- Tranditional physical layer DSP experts can reply on Commplax to boost their algorithms, optimize the most complicated parts, and further learn how deep learning works from bottom (autograd, optimizers, backprop, ...) to top (all kinds of layers, structures,....).
- ML researchers can play with Commplax to see the domain specfic parts is communication world (e.g. non-stationary random distortions, fiber non-liearties) and include Commplax's DSP operation as one of their toolbox to improve training capability


## Quickstart
The best way to get started with Commplax is through Jupyter's notebook demo, here are some examples
- [Hello world](https://github.com/remifan/commplax/examples/hello_world.ipynb) - demodulate DP-16QAM 815km SSMF signal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/remifan/commplax/blob/master/examples/hello_world.ipynb)
- (In progress) First glance of optimzation - optimize Digital Back Propogation (namely DNN-DBP or LDBP) while adapting DSP
- (In progress) Play with DNN - integrate TRAX

## Installation
PyPI package will be available after first release

run following steps in Python virtual enviroment (e.g. conda, pipenv) is strongly recommended

### Try commplax
- follow [JAX](https://github.com/google/jax)'s guide to install JAX
- `pip install https://github.com/remifan/commplax/archive/master.zip`

### Install for development
- follow [JAX](https://github.com/google/jax)'s guide to install JAX
- `git clone https://github.com/remifan/commplax && cd commplax`
  `pip install -e '.'`

## Where to get help
Commplax's is now under heavily development, any APIs might be changed constantly. It is encouraged to raise [Issue](https://github.com/remifan/commplax/issues) or join the [Dicussion panel](https://github.com/remifan/commplax/discussions)

## Open dataset for benchmarks
prepared in progress, will be ready soon

## Citing Commplax
Work in progress

## Reference documentation
Work in progress

## Commplax is created by
[Remi Qirui FAN](mailto:remi.qr.fan@gmail.com)

## Lovely people
- JAX community
- [Alan Pak Tao Lau](https://www.alanptlau.org/) (support) 


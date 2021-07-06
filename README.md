# Commplax: differentiable DSP for optical communication
[Documentation](https://commplax.readthedocs.io) |
[Chat](https://gitter.im/commplax/community)


Commplax is a modern Python DSP library mostly written in [JAX](https://github.com/google/jax), which is made by Google for high-performance machine learning research. Thanks to JAX's friendly API (most are [Numpy](https://numpy.org/)'s), efficient Autograd function and hardware acceleration, Commplax is/can:

- deal with Complex number well, thanks to JAX's native Complex number support
- shipped with accelerated well-tested DSP algorithms and core operations
- optimize computationally complex DSP (e.g. Digital Back Propogation) which is tranditionally inconvenient to do
- optimize DSP with deep learning models written in JAX's derivitives (e.g. [Flax](https://github.com/google/flax))
- designed carefully to maximize the readlibity and usability of the codebase
- flawlessly deploy to cloud runtime (e.g. [Colab](https://colab.research.google.com/), [Binder](https://mybinder.org/)) to share and colabrate

Commplax is designed for researchers in (optical) communication community and machine learning community, and hopefully may help to ease the collaboration between 2 worlds.
- Tranditional physical layer DSP experts can reply on Commplax to boost their algorithms, optimize the most complicated parts, and further learn how deep learning works from bottom (autograd, optimizers, backprop, ...) to top (all kinds of layers, network structures,....).
- ML researchers can play with Commplax to see the domain specfic parts in communication world (e.g. non-stationary random distortions, fiber non-liearties) and include Commplax's DSP operation as one of their toolbox to improve training capability


## Quickstart
The best way to get started with Commplax is through Jupyter's notebook demo, here are some examples
- [Hello world](https://github.com/remifan/commplax/blob/master/examples/hello_world.ipynb) - demodulate DP-16QAM 815km SSMF signal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/remifan/commplax/blob/master/examples/hello_world.ipynb)
- (In progress) First glance of optimzation - optimize Digital Back Propogation (namely DNN-DBP or LDBP) while adapting DSP
- (In progress) work with general DNN - integrate Flax

## Installation
PyPI package is not available yet

it is recommended to install in Python virtual enviroment (e.g. conda, pipenv).

### Quick install
Note jaxlib has no Windows builds yet, Windows 10+ users can use Commplax via [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about).

see [JAX](https://github.com/google/jax#installation) for more installation details.

_commplax is tested on Python 3.8, jax-0.2.13, jaxlib-0.1.68_
#### install CPU-only version
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```
#### install CPU+GPU version
you must install jaxlib that matches your cuda driver version, for example, if your CUDA version is 11.0,
```
pip install --upgrade jax==0.2.13 jaxlib==0.1.66+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

after jaxlib+cuda is installed,
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```

#### install TPU version
Cloud TPU VM jaxlib seems available, it is possible to run commplax in TPU. Will study TPU backend soon.

### Development
- follow [JAX](https://github.com/google/jax)'s guide to install JAX-CPU/GPU
- `git clone https://github.com/remifan/commplax && cd commplax`
  `pip install -e '.'`

## Where to get help
Commplax's is now under heavy development, any APIs might be changed constantly. It is encouraged to
- raise [Issue](https://github.com/remifan/commplax/issues)
- [Chat](https://gitter.im/commplax/community)
- [Dicussion panel](https://github.com/remifan/commplax/discussions)

## Open datasets for benchmarks
- [LabPtPTm1](https://github.com/remifan/LabPtPTm1)
- [LabPtPTm2](https://github.com/remifan/LabPtPTm2)

## Citing Commplax
```
@software{commplax2021github,
  author = {Qirui Fan and Chao Lu and Alan Pak Tao Lau},
  title = {{Commplax}: differentiable {DSP} library for optical communication},
  url = {https://github.com/remifan/commplax},
  version = {0.1.1},
  year = {2021},
}
```

## Reference documentation
For details about the Commplax API, see the [reference documentation](https://commplax.readthedocs.io) (work in progess)

## Acknowledgement
- [JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax)
- [Alan Pak Tao Lau](https://www.alanptlau.org/)
- [Chao Lu](http://www.eie.polyu.edu.hk/~enluchao/)


# Commplax: differentiable DSP for optical communication
[![Documentation Status](https://readthedocs.org/projects/commplax/badge/?version=latest)](https://commplax.readthedocs.io/en/latest/?badge=latest)

[Documentation](https://commplax.readthedocs.io) |


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
- [Demodulation](https://github.com/remifan/commplax/blob/master/docs/tutorial/equalizers.ipynb) - demodulate DP-16QAM 815km SSMF signal [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/remifan/commplax/blob/master/docs/tutorial/equalizers.ipynb)
- [Optimzing stateful layers](https://github.com/remifan/commplax/blob/master/docs/tutorial/stateful_layer.ipynb) - train Digital Back Propogation (namely DNN-DBP or LDBP) with adaptive filter layers ([Research artical](https://remifan.github.io/gdbp_study/overview.html)). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/remifan/commplax/blob/master/docs/tutorial/stateful_layer.ipynb)
- (In progress) work with general DNN - integrate Flax

## Installation
PyPI package is not available yet

it is recommended to install in Python virtual enviroment (e.g. conda, pipenv).

### Quick install
#### Windows installtion
Note jaxlib has no Windows builds yet, Windows 10 users can use Commplax via [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about), however Cuda is not supported.

We have succesfully run Commplax/JAX/Cuda in WSL2 shipped with Windows 11, see [WSL-Cuda](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for setup details. However the WSL-Cuda driver keeps randomly freezing our system and only certain versions of JAX are working well.

#### install CPU-only version
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```
#### install CPU+GPU version
install latest jax-cuda,
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```
you must install jaxlib that is compatible to your Cuda toolkit version,
```
# Installs the wheel compatible with Cuda 11 and cudnn 8.2 or newer.
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
see [JAX Installtion](https://github.com/google/jax#installation) for more information.

after jaxlib+cuda is installed,
```
pip install --upgrade https://github.com/remifan/commplax/archive/master.zip
```

### Development
- follow [JAX](https://github.com/google/jax)'s guide to install JAX-CPU/GPU
- `git clone https://github.com/remifan/commplax && cd commplax`
  `pip install -e '.'`

## Where to get help
Commplax's is now under heavy development, any APIs might be changed constantly. It is encouraged to
- [Issue](https://github.com/remifan/commplax/issues)
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


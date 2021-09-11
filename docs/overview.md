# Overview

## What is commplax
In short, commplax is a DSP library for optical communications written in Python, which has the following features:
- differentiable
- extremely fast by accelerations
- shipped with well-implemented operations and algorithms
- easy integration with state-of-the-art research ideas

Some basic experience in Python, [Numpy](https://numpy.org/) and [Jax](https://github.com/google/jax) are required to get started. There are many online references. Here are some that cover the basics

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [JAX for the Impatient](https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html)

## How can commplax help you
Commplax is made for researchers in optical communications by those who have been through the learning process. Since its very begining, commplax has been designed for both researchers and students. We hope we could help new students out since we understand most of the struggles to bridge the gaps from typical Matlab simulations to Python/Jax/Commplax

The traditional way of learning optical communications DSP is from bottom to top. Typically, one starts from learning the optics and communication theories in the textbooks. The next steps are self-learning and coding some known algorithms from published papers, lab instruments, and programming their toolbox. At certain point, one becomes knowledgable in some part of the DSP chain but still likely needs to reach out to his peers to help on the rest. But as time goes on, there will be more DSP algorithms to learn before one can propose something new for publishing. In addition, not just the papers but the way to present/reproduce the results should also evolve with time. In some senses, it is meaningful to not "just wander around looking at the stars but also look at the ground to fix the pothole before anyone falls in".

It is worthy to point out that research patterns in optical communications is in fact similar to that of machine learning in that both need datasets and codes for benchmarking. In ML research nowadays, one often starts with numerous public tutorials with example data and codes, there are public competitions over a specific dataset to source optimal solutions. Such accesibility may in turn help the whole community grow and expand faster.

As part of our preliminary efforts, we also open-sourced datasets on top of commplax to complete this dataset-codes loop. Users can run commplax with these datasets in their web browser through interactive notebooks we prepared and can build their new DSP from this platform for benchmarking.

## The current status and roadmap
Commplax is a open-sourced research project launched in Spring 2021 and has since been in active development. Most APIs are stable now but some documentations are still lacking.

- [x] core Jax-backed operations (Conv., Corr., FrFT, OverlapAdd,...)
- [x] common communication sources (random source, symbol de/encoding, pulse shaping...)
- [x] common equalizers and algorithms (CMA, LMS, Kalman...)
- [x] layer abstractions and gradient propagation of each algorithms/equalizers
- [x] Complex value compatible optimizers
- [x] online datasets with with programmable access
- [ ] synchronized documentation
- [ ] examples covering most common use cases
- [ ] integrating ML library, composing commplax's layers and layers from other lib like Flax
- [ ] transmission channel models
- [ ] incoperating implicit differentiation to support broader optimizations (with Reinforcement learning)

## What does the workflow of commplax look like
```{thebe-button} Click Here First to Activate Interaction!
```
The activation process may take a while. After the clicked button turns <span style="color:green">**ready**</span>, click the "run" button of the codeblock below, which may take several seconds to install the dependencies. Variables and their status are shared across codeblocks, once the current run gets finished you can move to the next codeblock and repeat this 'run-wait-result' loop. 

### Install commplax and example dataAPI


```{code-block}
:class: thebe

print('installing...')
%pip install --upgrade --quiet https://github.com/remifan/commplax/archive/master.zip
%pip install --upgrade --quiet https://github.com/remifan/LabPtPTm1/archive/master.zip
print('done.')
```

### Work the example
```{code-block}
:class: thebe

import numpy as np
import matplotlib.pyplot as plt
from labptptm1 import dataset as dat
from commplax import comm, xcomm, equalizer as eq, plot as cplt

print('downloading data...')
ds = dat['815km_SSMF/DP16QAM_RRC0.2_28GBd_1ch']
y = ds['LP-6_5/recv'][:100000]

sr = ds.attrs['samplerate']
br = ds.attrs['baudrate']
dist = ds.attrs['distance']
spans = ds.attrs['spans']
mf = ds.attrs['modformat']
lpw = ds['LP-6_5'].attrs['lpw']
CD = 13.6 # s/m
         
print('shape of y: %s (received waveforms, resampled to 2 samples/symbol): ' % str(y.shape))
print('sample rate: %.1f GHz' % (sr / 1e9))
print('baud rate: %.1f GBd' % (br / 1e9))
print('launched power: %.3f mW' % (lpw * 1e3))
print('link distance: %.1f km (measured)' % (dist / 1e3))
print('number of spans: %d' % spans)
print('done.')
```

```{code-block}
:class: thebe

# DSP chain
y =  xcomm.normpower(y - np.mean(y, axis=0), real=True) / np.sqrt(2) # remove DC and normalize signal
z = eq.cdcomp(y, sr, CD) # chromatic disperison compensation
z = eq.modulusmimo(z, taps=19, lr=2**-14)[0]  # polarization demux
z = eq.qamfoe(z)[0] # frequency offset equalization
z = eq.ekfcpr(z)[0] # finer carrier phase recovery

cplt.scatter(z[40000:45000])
print('done.')
```

This handy interactive run serves as a preview of [Jupyter Notebook](https://jupyter.org/), which you will be most likely to live with through the whole documentation.

See [Equalizers](https://commplax.readthedocs.io/en/latest/tutorial/equalizers.html) for full version of the above example.


## How to use this document
The documentation is lagging much behind the codebase at this moment, we are actively building the [Tutorial](https://commplax.readthedocs.io/en/latest/tutorial/index.html), of which each page is executable in cloud runtimes by clicking the badges at the top: ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg), ![Open In Mybinderorg](https://mybinder.org/badge_logo.svg). Since the cloud storage is often not persistent, for long-term use, it is suggested to follow the [Installation](https://commplax.readthedocs.io/en/latest/installation.html) to setup your local environment.

You may refer to the [Public APIs](https://commplax.readthedocs.io/en/latest/commplax.html) for interface details.
from flax.core import Scope, init, apply
from jax import numpy as jnp, random, grad, jit
from commplax import comm, xcomm, xop, cxopt, adaptive_filter as af
import numpy as np
from typing import NamedTuple, Callable, Optional, Any


Array = Any


class Signal(NamedTuple):
  x: Array
  t: Array


def zeros(key, shape, dtype=jnp.float32): return jnp.zeros(shape, dtype)
def delta(key, shape, dtype=jnp.float32):
    k1d = comm.delta(shape[0], dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])
def gauss(key, shape, dtype=jnp.dtype):
    taps = shape[0]
    k1d = comm.gauss(comm.gauss_minbw(taps), taps=taps, dtype=dtype)
    return jnp.tile(np.expand_dims(k1d, axis=list(range(1, len(shape)))), (1,) + shape[1:])


def conv1d_taxis(t, taps, rtap, stride, mode):
    if rtap is None:
        rtap = (taps - 1) // 2
    delay = -(-(rtap + 1) // stride) - 1
    if mode == 'full':
        tslice = (-delay * stride, taps - stride * (rtap + 1)) #TODO: think more about this
    elif mode == 'same':
        tslice = (0, 0)
    elif mode == 'valid':
        tslice = (delay * stride, (delay + 1) * stride - taps)
    else:
        raise ValueError('invalid mode {}'.format(mode))
    return t[tslice[0]: t.shape[0] + tslice[1]: stride]


def conv1d(
    scope: Scope,
    inputs,
    taps=31,
    rtap=None,
    mode='valid',
    kernel_init=zeros,
    conv_fn = xop.convolve):

    x, t = inputs
    t = conv1d_taxis(t, taps, rtap, 1, mode=mode)
    h = scope.param('kernel',
                     kernel_init,
                     (taps,), np.complex64)
    x = conv_fn(x, h, mode=mode)

    return Signal(x, t)


def mimoconv1d(
    scope: Scope,
    inputs,
    taps=31,
    rtap=None,
    dims=2,
    mode='valid',
    kernel_init=zeros,
    conv_fn=xop.convolve):

    x, t = inputs
    t = conv1d_taxis(t, taps, rtap, 1, mode=mode)
    h = scope.param('kernel', kernel_init, (taps, dims, dims), np.float32)
    y = xcomm.mimoconv(x, h, mode=mode, conv=conv_fn)
    return Signal(y, t)


def mimoaf(
    scope: Scope,
    inputs,
    truth=None,
    taps=32,
    rtap=None,
    dims=2,
    sps=2,
    train=False,
    mimofn=af.ddlms,
    mimokwargs={},
    mimoinitargs={}):

    x, t = inputs
    x = xop.frame(x, taps, sps)
    mimo_init, mimo_update, mimo_apply = mimofn(train=train, **mimokwargs)
    t = conv1d_taxis(t, taps, rtap, sps, mode='valid')
    if truth is not None:
        truth = jnp.take(truth, t.astype(int))
    state = scope.variable('af_state', 'mimo_state',
                           lambda *_: (0, mimo_init(dims=dims, taps=taps, **mimoinitargs)), ())
    af_step, af_stats = state.value
    af_step, (af_stats, (af_weights, _)) = af.iterate(mimo_update, af_step, af_stats, x, truth)
    y = mimo_apply(af_weights, x)
    state.value = (af_step + x.shape[0],) + (af_stats,)
    return Signal(y, t)


def fdbp(
    scope: Scope,
    inputs,
    steps=3,
    dtaps=261,
    ntaps=41,
    sps=2,
    d_init=delta,
    n_init=gauss):

    x, t = inputs

    for i in range(steps):
        x0, t0 = scope.child(conv1d, name='Dx_%d' % i)(Signal(x[:, 0], t[:, 0]), taps=dtaps, kernel_init=d_init)
        x1, t1 = scope.child(conv1d, name='Dy_%d' % i)(Signal(x[:, 1], t[:, 1]), taps=dtaps, kernel_init=d_init)
        x = jnp.stack([x0, x1], axis=-1)
        t = jnp.stack([t0, t1], axis=-1)
        c, t = scope.child(mimoconv1d, name='N_%d' % i)(Signal(jnp.abs(x)**2, t), taps=ntaps, kernel_init=n_init)
        x = jnp.take(x, (t * sps).astype(int)) * jnp.exp(1j * c)

    return Signal(x, t)


def serial(*fs):
    def _serial(scope, inputs, **kwargs):
        truth = kwargs.pop('truth', None)
        for f in fs:
            try:
                inputs = scope.child(f)(inputs, truth=truth, **kwargs)
            except TypeError:
                inputs = scope.child(f)(inputs, **kwargs)
        return inputs
    return _serial


def parallel(*fs):
    def _parallel(scope, inputs, **kwargs):
        truth = kwargs.pop('truth', None)
        outputs = []
        for f, inp in zip(fs, inputs):
            try:
                out = scope.child(f)(inp, truth=truth, **kwargs)
            except TypeError:
                out = scope.child(f)(inp, **kwargs)
            outputs.append(out)

        return outputs
    return _parallel



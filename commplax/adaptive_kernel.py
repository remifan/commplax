# Copyright 2026 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses as dc
import functools
import numpy as np
import jax
from typing import Any, TypeVar, Callable, Optional, Tuple, Union
from functools import partial
from jax import numpy as jnp, vmap, jit, lax
from jax.tree_util import tree_flatten, tree_unflatten
from jax.lax import stop_gradient
from jaxtyping import Array, Int, Float, PyTree

from commplax import signal, sym_map
from commplax.optim import Schedule, make_schedule
from commplax.jax_util import default_complexing_dtype, astuple

Params = Any
State = PyTree
AFStep = int
AFInput = PyTree | Array
AFState = Any

InitFn = Callable[[Any], AFState]
UpdateFn = Callable[[AFStep, AFState, AFInput], PyTree]
ApplyFn = Callable[[AFState, AFInput], PyTree]

@dc.dataclass
class AdaptiveFilter():
    init: InitFn = None
    update: UpdateFn = None
    apply: ApplyFn = None

    def __iter__(self):
        return iter((self.init, self.update, self.apply))


C = TypeVar('C', bound=Callable)


def adaptive_filter(af_maker: C, trainable: bool=False, stop_grad_on_update: bool=True) -> C:
    grad_handler = stop_gradient if stop_grad_on_update else lambda *args: args
    @functools.wraps(af_maker)
    def _af_maker(*args, **kwargs):
        af_made = af_maker(*args, **kwargs)
        init, update, apply = af_made

        @functools.wraps(init)
        def _init(*args, **kwargs):
            x0 = init(*args, **kwargs)
            return jax.device_put(x0)

        @functools.wraps(update)
        def _update(i, af_state, af_inp):
            if trainable:
                af_inp = af_inp if isinstance(af_inp, tuple) else (af_inp,)
                dims = af_inp[0].shape[-1]
                dummy_zeros = jnp.zeros(dims)
                af_inp = (af_inp + (dummy_zeros,))[:2] # stuff dummy values in absence of ground truth
            else:
                af_inp = af_inp[0] if isinstance(af_inp, tuple) else af_inp
            af_inp = jax.device_put(af_inp) # uncommited
            af_state = jax.device_put(af_state) # uncommited
            af_state, af_out = grad_handler(update(i, af_state, af_inp))
            return af_state, af_out

        @functools.wraps(apply)
        def _apply(af_ps, af_xs):
            return apply(af_ps, af_xs)

        _update.trainable = trainable
        _update.train = getattr(update, 'train', make_schedule(False))

        return AdaptiveFilter(_init, _update, _apply)

    return _af_maker


def array(af_maker, replicas, axis=-1):
    @functools.wraps(af_maker)
    def rep_af_maker(*args, **kwargs):
        init, update, apply = af_maker(*args, **kwargs)

        @functools.wraps(init)
        def rep_init(*args, **kwargs):
            x0 = init(*args, **kwargs)
            x0_flat, x0_tree = tree_flatten(x0)
            x0_flat = tuple(map(lambda v: jnp.repeat(v[..., None], replicas, axis=axis), x0_flat))
            x0 = tree_unflatten(x0_tree, x0_flat)
            return x0

        # @jax.jit
        @functools.wraps(update)
        def rep_update(i, af_state, af_inp):
            af_state, af_out = jax.vmap(update, in_axes=(None, axis, axis), out_axes=axis)(i, af_state, af_inp)
            return af_state, af_out

        # @jax.jit
        @functools.wraps(apply)
        def rep_apply(af_ps, af_xs):
            return jax.vmap(apply, in_axes=axis, out_axes=axis)(af_ps, af_xs)

        return AdaptiveFilter(rep_init, rep_update, rep_apply)

    return rep_af_maker


def dtype_to_real(rvs_dtype):
    return rvs_dtype.type(0).real.dtype


def dtype_to_complex(rvs_dtype):
    return (1j * rvs_dtype.type(0)).dtype


def frame(y, taps, sps, rtap=None):
    y_pad = jnp.pad(y, mimozerodelaypads(taps=taps, sps=sps, rtap=rtap))
    yf = jnp.array(signal.frame(y_pad, taps, sps))
    return yf


def iterate(update: UpdateFn,
            state: AFState,
            signal: AFInput,
            truth=None,
            # truth_ndim=2,
            step0: AFStep=None,
            device=None):
    _step0 = 0 if step0 is None else step0
    steps = _step0 + jnp.arange(signal.shape[0])
    # pad dummy truth
    truth_ndim = signal.ndim if truth is None else truth.ndim
    truth = jnp.zeros((0, *signal.shape[1 - truth_ndim:]), dtype=signal.dtype) if truth is None else truth[:signal.shape[0]]
    padw_data_axes = ((0, 0),) * (truth_ndim - 1)
    truth = jnp.pad(truth, ((0, signal.shape[0] - truth.shape[0]), *padw_data_axes))
    xs = (steps, signal, truth)
    res = signal.scan(lambda c, xs: update(xs[0], c, xs[1:]), state, xs, jit_device=device)
    out = res if step0 is None else (steps[-1], res)
    return out


def mimo(w, u):
    return jnp.einsum('ijt,tj->i', w, u)


def r2c(r):
    ''' Pack real-valued signal into complex-valued signal
    for example, converting
    [0.  0.  1. -1.]
    to
    [0.+ 0.j 1.-1.j]
    '''
    if not jnp.iscomplexobj(r):
        if r.ndim != 1:
            raise ValueError('invalid ndim, expected 1 but got %d' % r.ndim)
        I = np.arange(r.shape[0]//2) * 2
        c  = r[I] + 1j*r[I+1]
    else:
        c = r
    return c


def c2r(c):
    ''' Unpack complex-valued signal into real-valued signal
    for example, converting
    [0.+0.j 1.-1.j]
    to
    [ 0. 0. 1. -1.]
    '''
    if jnp.iscomplexobj(c):
        if c.ndim != 1:
            raise ValueError('invalid ndim, expected 1 but got %d' % c.ndim)
        dims = c.shape[0]
        r = jnp.empty(shape=(dims * 2,), dtype=dtype_to_real(c.dtype))
        I = np.arange(dims)
        r = r.at[I*2+0].set(c.real)
        r = r.at[I*2+1].set(c.imag)
    else:
        r = c
    return r


def c2r_mimo_weights(wc):
    dims, taps = wc.shape[0], wc.shape[2]
    wr = jnp.empty(shape=(dims * 2, dims * 2, taps), dtype=dtype_to_real(wc.dtype))
    I = np.arange(dims)
    wr = wr.at[*np.meshgrid(I*2+0, I*2+0, indexing='ij'), :].set( wc.real)
    wr = wr.at[*np.meshgrid(I*2+0, I*2+1, indexing='ij'), :].set(-wc.imag)
    wr = wr.at[*np.meshgrid(I*2+1, I*2+0, indexing='ij'), :].set( wc.imag)
    wr = wr.at[*np.meshgrid(I*2+1, I*2+1, indexing='ij'), :].set( wc.real)
    return wr

vc2r = vmap(c2r)
vr2c = vmap(r2c)
vmimo = vmap(mimo)

def filterzerodelaypads(taps, stride=1, rtap=None):
    if rtap is None:
        rtap = (taps + 1) // 2 - 1
    filterdelay = int(np.ceil((rtap + 1) / stride) - 1)
    pads = np.array([[filterdelay * stride, taps - stride * (filterdelay + 1)], [0,0]])
    return pads


def mimozerodelaypads(taps, sps=2, rtap=None):
    return filterzerodelaypads(taps, sps, rtap)


def mimoinitializer(taps, dims, dtype=None, initkind='centralspike', nspike=1):
    dtype = default_complexing_dtype() if dtype is None else dtype
    if np.isscalar(taps):
        match initkind.lower():
            case "zeros":
                w0 = jnp.zeros((dims, dims, taps), dtype=dtype)
            case "centralspike":
                w0 = jnp.zeros((dims, dims, taps), dtype=dtype)
                ctap = (taps + 1) // 2 - 1 
                ctaps = jnp.arange(ctap - nspike//2, ctap + (nspike+1)//2)
                for i in ctaps:
                    w0 = w0.at[np.arange(dims), np.arange(dims), i].set(1.)
            case _:
                raise ValueError('invalid initkind %s' % initkind)
    else:
        w0 = jnp.asarray(taps, dtype=dtype)
    return w0


def decision(const, v, stopgrad=True):
    """ simple symbol decision based on Euclidean distance
    """
    if v.ndim > 1:
        raise ValueError(f'ndim = 1 is expected, but got {v.ndim} instead')
    v = jnp.atleast_1d(v)
    i = jnp.argmin(jnp.abs(const[:, None] - v[None, :]), axis=0)
    d = const[i]
    return stop_gradient(d) if stopgrad else d


def partition_QAM(x, slicers=None):
    is_scalar = jnp.isscalar(x)
    x = jnp.atleast_1d(x)
    if slicers is None:
        radii = jnp.sqrt(jnp.array([2, 10, 18]) / 10) # normalized 16-QAM
        slicers = (radii[1:] + radii[:-1]) / 2
    groups = jnp.sum(jnp.abs(x)[:, None] > slicers[None, :], axis=1)
    groups = jnp.squeeze(groups) if is_scalar else groups
    return groups


@partial(adaptive_filter, trainable=True)
def lms(
    lr: Union[float, Schedule] = 1e-3,
    train: Union[bool, Schedule] = False,
    const: Optional[Array]=None,
    norm: bool=True,
    tap_leakage: float = 0.,
    bias: bool = False,
    bias_lr: Optional[Union[float, Schedule]] = None,
) -> AdaptiveFilter:
    """LMS MIMO adaptive filter.

    Args:
      lr: Learning rate for filter taps
      train: Training mode schedule (uses ground truth when True)
      const: Constellation for decision-directed mode (default: 16QAM)
      norm: If True, use normalized LMS
      tap_leakage: Tap leakage factor for regularization
      bias: If True, include adaptive bias term for slow-varying DC offset
      bias_lr: Learning rate for bias (default: lr/10 for slower adaptation)

    Returns:
      an ``AdaptiveFilter`` object

    Note:
      When bias=True, the state becomes (w, b) where b is the bias vector.
      The bias is updated with a slower learning rate to track DC offset.
    """
    α = make_schedule(lr)
    α_bias = make_schedule(bias_lr if bias_lr is not None else lr * 0.1)
    β = tap_leakage
    σ = 1e-5
    train = make_schedule(train)
    const = jnp.asarray(sym_map.const('16QAM', norm=True)) if const is None else const

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike')
        if bias:
            b0 = jnp.zeros(dims, dtype=dtype)
            s0 = w0, b0
        else:
            s0 = w0,
        return s0

    def loss_fn_no_bias(w, inp, i):
        u, x = inp
        v = mimo(w, u)
        d = jnp.where(train(i), x, decision(const, v))
        loss = jnp.sum(jnp.abs(d - v)**2)
        return loss

    def loss_fn_with_bias(params, inp, i):
        w, b = params
        u, x = inp
        v = mimo(w, u) + b
        d = jnp.where(train(i), x, decision(const, v))
        loss = jnp.sum(jnp.abs(d - v)**2)
        return loss

    def update(i, s, inp):
        u = inp[0]
        if bias:
            w, b = s
            l, (g_w, g_b) = jax.value_and_grad(loss_fn_with_bias)((w, b), inp, i)
            out = (w, b, l)
            if norm:
                μ = α(i) * 1 / (σ + jnp.sum(jnp.abs(u)**2, axis=0))[:, None, None]
            else:
                μ = α(i)
            w = (1-μ*β) * w - μ * g_w.conj()
            b = b - α_bias(i) * g_b.conj()
            s = w, b
        else:
            w, = s
            l, g = jax.value_and_grad(loss_fn_no_bias)(w, inp, i)
            out = (w, l)
            if norm:
                μ = α(i) * 1 / (σ + jnp.sum(jnp.abs(u)**2, axis=0))[:, None, None]
            else:
                μ = α(i)
            w = (1-μ*β) * w - μ * g.conj()
            s = w,
        return s, out

    def apply(s, y):
        if bias:
            w, b = s
            return mimo(w, y) + b
        else:
            w = tuple(s)[0]
            return mimo(w, y)

    update.train = train

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def lms_block(
    lr: Union[float, Schedule] = 1e-3,
    train: Union[bool, Schedule] = False,
    const: Optional[Array] = None,
    norm: bool = True,
    tap_leakage: float = 0.,
) -> AdaptiveFilter:
    """Block-based LMS MIMO adaptive filter.

    Processes N symbols per update, computing gradient over the entire block.
    More efficient on GPU/TPU than symbol-by-symbol LMS.

    Args:
        lr: Learning rate (scalar or Schedule)
        train: Training mode schedule (uses ground truth when True)
        const: Constellation for decision-directed mode
        norm: If True, use normalized LMS
        tap_leakage: Tap leakage factor for regularization

    Returns:
        an ``AdaptiveFilter`` object

    Note:
        Input shape per update: (N, taps, dims) where N is block size.
        Converges similarly to symbol-by-symbol LMS but with one update per block.
    """
    α = make_schedule(lr)
    β = tap_leakage
    σ = 1e-5
    train = make_schedule(train)
    const = jnp.asarray(sym_map.const('16QAM', norm=True)) if const is None else const

    # Batched MIMO: process N frames at once
    mimo_block = jax.vmap(mimo, in_axes=(None, 0))

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike)
        s0 = w0,
        return s0

    def loss_fn(w, inp, i):
        u_block, x_block = inp  # u_block: (N, taps, dims), x_block: (N, dims)
        v_block = mimo_block(w, u_block)  # (N, dims)
        d_block = jnp.where(train(i), x_block, vmap(lambda v: decision(const, v))(v_block))
        loss = jnp.mean(jnp.sum(jnp.abs(d_block - v_block)**2, axis=-1))
        return loss

    def update(i, s, inp):
        w, = s
        u_block = inp[0]  # (N, taps, dims)
        l, g = jax.value_and_grad(loss_fn)(w, inp, i)
        out = (w, l)
        if norm:
            # Average normalization over the block
            u_power = jnp.mean(jnp.sum(jnp.abs(u_block)**2, axis=1), axis=0)  # (dims,)
            μ = α(i) * 1 / (σ + u_power)[:, None, None]
        else:
            μ = α(i)
        w = (1 - μ * β) * w - μ * g.conj()
        s = w,
        return s, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    update.train = train

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def rls_lms(
    λ: Union[float, Schedule] = 0.999,
    δ: float = 0.01,
    train: Union[bool, Schedule] = False,
    const: Optional[Array]=None,
    bias: bool = False,
    bias_lr: Union[float, Schedule] = 1e-4,
) -> AdaptiveFilter:
    """RLS-based LMS MIMO adaptive filter.

    Args:
      λ: Forgetting factor (default: 0.999)
      δ: Regularization for initial P matrix
      train: Training mode schedule (uses ground truth when True)
      const: Constellation for decision-directed mode (default: 16QAM)
      bias: If True, include adaptive bias term for slow-varying DC offset
      bias_lr: Learning rate for bias (default: 1e-4 for slow adaptation)

    Returns:
      an ``AdaptiveFilter`` object

    Note:
      When bias=True, the state becomes (w, P, b) where b is the bias vector.
      The bias is updated with gradient descent (separate from RLS tap update).
    """
    _λ = make_schedule(λ)
    _bias_lr = make_schedule(bias_lr)
    train = make_schedule(train)
    const = jnp.asarray(sym_map.const('16QAM', norm=True)) if const is None else const

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike)
        P0 = jnp.tile(δ * jnp.eye(taps * dims, dtype=dtype), (dims, 1, 1))
        if bias:
            b0 = jnp.zeros(dims, dtype=dtype)
            s0 = (w0, P0, b0)
        else:
            s0 = (w0, P0)
        return s0

    @partial(jax.vmap, in_axes=(None, 0, (None, 0)), out_axes=(0, -1))
    def update_no_bias(i, s, inp):
        w, _P = s
        u, x = inp
        N, dims = u.shape[0], w.shape[0]
        # in case of polyphase filtering, P is cropped
        i_P = jnp.ix_(jnp.arange(N*dims), jnp.arange(N*dims))
        P = _P[i_P]

        u_i = u.reshape(-1, order='F')[:, None]
        h = w.conj().reshape(-1)[:, None]
        λ = _λ(i)

        k = 1/λ * P @ u_i / (1 + 1/λ * u_i.conj().T @ P @ u_i)
        v = jnp.squeeze(h.conj().T @ u_i)
        d = jnp.where(train(i), x, decision(const, v))
        ε = d - v
        h = h + k * ε.conj()
        P = 1/λ * P - 1/λ * k * u_i.conj().T @ P

        dims = w.shape[0]
        w = h.conj().reshape((dims, -1))

        _P = _P.at[i_P].set(P)
        s = w, _P
        out = (ε, v)
        return s, out

    @partial(jax.vmap, in_axes=(None, 0, (None, 0)), out_axes=(0, -1))
    def update_with_bias(i, s, inp):
        w, _P, b = s
        u, x = inp
        N, dims = u.shape[0], w.shape[0]
        # in case of polyphase filtering, P is cropped
        i_P = jnp.ix_(jnp.arange(N*dims), jnp.arange(N*dims))
        P = _P[i_P]

        u_i = u.reshape(-1, order='F')[:, None]
        h = w.conj().reshape(-1)[:, None]
        λ = _λ(i)

        k = 1/λ * P @ u_i / (1 + 1/λ * u_i.conj().T @ P @ u_i)
        v = jnp.squeeze(h.conj().T @ u_i) + b
        d = jnp.where(train(i), x, decision(const, v))
        ε = jnp.squeeze(d - v)
        h = h + k * ε.conj()
        P = 1/λ * P - 1/λ * k * u_i.conj().T @ P
        # Update bias with simple gradient descent (ensure scalar output for vmap)
        b = jnp.squeeze(b + _bias_lr(i) * ε.conj())

        dims = w.shape[0]
        w = h.conj().reshape((dims, -1))

        _P = _P.at[i_P].set(P)
        s = w, _P, b
        out = (ε, v)
        return s, out

    def update(i, s, inp):
        if bias:
            return update_with_bias(i, s, inp)
        else:
            return update_no_bias(i, s, inp)

    def apply(s, y):
        w = astuple(s)[0]
        v = mimo(w, y)
        if bias:
            b = s[2]
            return v + b
        return v

    update.train = train

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def lms_MoriY(
    lr_w: Union[float, Schedule] = 1/2**5,
    lr_f: Union[float, Schedule] = 1/2**6,
    lr_s: Union[float, Schedule] = 0.,
    lr_b: Union[float, Schedule] = 1/2**10,
    train: Union[bool, Schedule] = False,
    grad_max: Tuple[float, float] = (30., 30.),
    eps: float = 1e-8,
    beta: float = 0.,
    const: Array = sym_map.const("16QAM", norm=True)
) -> AdaptiveFilter:
    """Decision-Directed Least Mean Square adaptive equalizer

    Args:
      lr_w: learning rate of MIMO(butterfly part)'s weights
      lr_f: learning rate of stage-I phase tracker
      lr_s: learning rate of stage-II phase tracker
      lr_b: learning rate of bias term
      train: controlling flag of training mode, which can be a bool for global control within one call
        or an array of bool to swich training on iteration basis
      grad_max: clipling threshold of the gradients of phase trackers
      eps: perturbative term to stablize normalized LMS
      beta: smoothening factor of phase trackers
      const: Optional; constellation used to infer R2 when R2 is None

    Returns:
      an ``AdaptiveFilter`` object

    Notes:
      - add bias term to handle varying DC component

    References:
      - [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of
        finite-impulse-response filters tolerant to carrier-phase fluctuations
        in digital coherent optical receivers for higher-order quadrature
        amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    """
    const = jnp.asarray(const)
    lr_w = make_schedule(lr_w)
    lr_f = make_schedule(lr_f)
    lr_s = make_schedule(lr_s)
    lr_b = make_schedule(lr_b)
    train = make_schedule(train)

    def init(taps=32, dims=2, dtype=default_complexing_dtype(), nspike=1, w0=None):
        w0 = mimoinitializer(taps, dims, dtype, 'zeros') if w0 is None else w0
        cplx = default_complexing_dtype()
        _dims = dims if jnp.iscomplexobj(w0) else dims // 2
        f0 = jnp.full((_dims,), 1., dtype=cplx)
        s0 = jnp.full((_dims,), 1., dtype=cplx)
        b0 = jnp.full((_dims,), 0., dtype=cplx)
        fshat0 = jnp.full((_dims,), 1., dtype=cplx)
        return (w0, f0, s0, b0, fshat0)

    def update(i, state, inp):
        w, f, s, b, fshat = state
        u, x = inp

        v = r2c(mimo(w, u))
        k = v * f
        c = k * s
        z = c + b
        q = v * fshat + b
        d = jnp.where(train(i), r2c(x), decision(const, q))
        l = jnp.sum(jnp.abs(z - d)**2)

        psi_hat = jnp.abs(f)/f * jnp.abs(s)/s
        e_w = (d - b) * psi_hat - v
        e_w = c2r(e_w) if not jnp.iscomplexobj(w) else e_w
        e_f = d - b - k
        e_s = d - b - c
        e_b = d - z
        gw = -1. / ((jnp.abs(u)**2).sum() + eps) * e_w[:, None, None] * u.conj().T[None, ...]

        gf = -1. / (jnp.abs(v)**2 + eps) * e_f * v.conj()
        gs = -1. / (jnp.abs(k)**2 + eps) * e_s * k.conj()
        gb = -e_b

        # bound the grads of f and s which are less regulated than w,
        # it may stablize this algo. by experience
        gf = jnp.where(jnp.abs(gf) > grad_max[0], gf / jnp.abs(gf) * grad_max[0], gf)
        gs = jnp.where(jnp.abs(gs) > grad_max[1], gs / jnp.abs(gs) * grad_max[1], gs)

        out = ((w, f, s, b), (l, d))

        # update
        w = w - lr_w(i) * gw
        f = f - lr_f(i) * gf
        s = s - lr_s(i) * gs
        b = b - lr_b(i) * gb
        fshat = beta * fshat + (1 - beta) * (f * s)

        state = (w, f, s, b, fshat)

        return state, out

    def apply(state, y):
        w, f, s, b = state[:4]
        res = r2c(mimo(w, y)) * f * s + b
        return c2r(res) if not jnp.iscomplexobj(w) else res

    update.train = train

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def cma(
    lr: Union[float, Schedule] = 0.03,
    R2: Optional[float]=1.32,
    const: Optional[Array]=None,
    norm: bool=True,
    leakage_params: tuple=(1., 4., 200, 1, 3),
):
    """CMA blind MIMO adaptive filter.

    Args:
      lr: Optional; learning rate
      R2: Optional; reference square of radius
      const: Optional; constellation used to infer R2 when R2 is None

    Returns:
      an ``AdaptiveFilter`` object

    Examples:
      >>> from commplax.adaptive_kernel import cma
      >>>
      >>> af = cma(lr=1e-4, R2=1.32) # for non-PS power normalized 16QAM signal

    Notes:
      leakage factor: prevent tap build-up and to provide out-of-band Spectral Shaping in order to
        reject the out-of-band noise and minimize adjacent channel interference.

    References:
      - [1] D. Godard, “Self-recovering equalization and carrier tracking in two-dimensional data
        communication systems,” IEEE Trans. Commun., vol. 28, no. 11, pp. 1867-1875, Nov. 1980.
      - [2] K. Kikuchi, “Polarization-demultiplexing algorithm in the digital coherent receiver,”
        in Proc. Digest 2008 IEEE/LEOS Summer Topical Meetings, Jul., pp. 101-102.
      - [3] Jones, Douglas L. "A normalized constant-modulus algorithm." Conference Record of the
        Twenty-Ninth Asilomar Conference on Signals, Systems and Computers. Vol. 1. IEEE, 1995.
      - [4] Kamenetsky, Max, and Bernard Widrow. "A variable leaky LMS adaptive algorithm." 
        Conference Record of the Thirty-Eighth Asilomar Conference on Signals, Systems and Computers,
        2004.. Vol. 1. IEEE, 2004.
    """
    α = make_schedule(lr)
    γ_max, β, M, lu, ld = leakage_params
    γ = lambda m: γ_max * jnp.power(m / M, β)
    σ = 1e-5

    if const is not None:
        R2 = jnp.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))
    R1 = jnp.sqrt(R2)

    def init(taps=19, dims=2, dtype=np.complex64, nspike=1, w0=None):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike) if w0 is None else w0
        m0 = 1 # small value of leak, see [4]
        s0 = (w0, m0)
        return s0

    def loss_fn(w, u):
        v = r2c(mimo(w, u))
        v2 = jnp.abs(v)**2
        loss = jnp.sum(jnp.abs(R2 - v2)**2)
        return loss

    def update(i, s, u):
        w, m = s
        l, g = jax.value_and_grad(loss_fn)(w, u)

        if norm: # see [3]
            # normalzied CMA
            v = mimo(w, u)
            v1 = jnp.abs(v)
            v2 = v1**2
            E = jnp.sum(jnp.abs(u)**2, axis=0)
            μ = α(i) * ((v2 - R1 * v1) / (4 * v2 * (v2 - R2) * E + σ))[:, None, None]
        else:
            μ = α(i)

        # compare the posteriori errors for leakage adjustment, see [4]
        _w = (1 - μ) * w - μ * g.conj()
        w = (1 - μ * γ(m)) * w - μ * g.conj()
        m = lax.select(
            loss_fn(w, u) < loss_fn(_w, u),
            lax.min(m + lu, M),
            lax.max(m - ld, 0),
        )

        s = (w, m)
        out = (w, l)
        return s, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def rls_cma(
    λ: Union[float, Schedule] = 0.999,
    δ: float = 0.01,
    R2: Optional[float]=1.32,
    const: Optional[Array]=None
) -> AdaptiveFilter:
    """RLS-CMA blind MIMO adaptive filter.
    λ: forgetting factor (λ<1)
    δ: is small positive number
    [1] Md. S. Faruk and S. J. Savory, “Digital Signal Processing for
     Coherent Transceivers Employing Multilevel Formats,” J. Lightwave
     Technol., vol. 35, no. 5, pp. 1125-1141, Mar. 2017, doi: 10.1109/JLT.2017.2662319.

    """
    _λ = make_schedule(λ)

    if const is not None:
        R2 = jnp.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike) if w0 is None else w0
        P0 = jnp.tile(δ * jnp.eye(taps * dims, dtype=dtype), (dims, 1, 1))
        return (w0, P0)

    @partial(jax.vmap, in_axes=(None, 0, None), out_axes=(0, -1))
    def update(i, s, u):
        w, _P = s
        N, dims = u.shape[0], w.shape[0]
        # in case of polyphase filtering, P is downsized
        i_P = jnp.ix_(jnp.arange(N*dims), jnp.arange(N*dims))
        P = _P[i_P]

        u_i = u.reshape(-1, order='F')[:, None]
        h = w.conj().reshape(-1)[:, None]
        λ = _λ(i)

        z = u_i @ u_i.conj().T @ h
        k = 1/λ * P @ z / (1 + 1/λ * z.conj().T @ P @ z)
        v = h.conj().T @ z
        ε = jnp.squeeze(R2 - v)
        h = h + k * ε.conj()
        P = 1/λ * P - 1/λ * k * z.conj().T @ P

        w = h.conj().reshape((dims, -1))

        _P = _P.at[i_P].set(P)
        s = w, _P
        out = (ε,)
        return s, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def mu_cma(
    lr: Union[float, Schedule] = 1e-4,
    R2: Union[float, Schedule] = 1.32,
    delta: int = 6,
    beta: float = 0.999,
    const: Optional[Array] = None
) -> AdaptiveFilter:
    """Multiuser CMA - Singularity-free blind MIMO equalizer
    
    Args:
      dims: dimension of input signal
      lr: learning rate
      R2: reference squared radius
      delta: the number of symbols used in evaluating the cross-correlation
      beta: smoothing factor of exponential moving everage
      const: Optional; constellation used to infer R2 when R2 is None
      
    Returns:
      an ``AdaptiveFilter`` object

    References:
      - [1] Papadias, Constantinos B., and Arogyaswami J. Paulraj. "A constant modulus algorithm
        for multiuser signal separation in presence of delay spread using antenna arrays."
        IEEE signal processing letters 4.6 (1997): 178-181.
      - [2] Vgenis, Athanasios, et al. "Nonsingular constant modulus equalizer for PDM-QPSK coherent
        optical receivers." IEEE Photonics Technology Letters 22.1 (2009): 45-47.
    """
    lr = make_schedule(lr)

    if const is not None:
        R2 = jnp.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike)
        z0 = jnp.zeros((delta, dims), dtype=dtype)
        r0 = jnp.zeros((dims, dims, delta), dtype=dtype)
        beta_ = jnp.asarray(beta)
        s0 = (w0, z0, r0, beta_)
        return s0 

    def update(i, state, u):
        dims = u.shape[1]
        w, z, r, betapow = state
        z = jnp.concatenate((mimo(w, u)[None, :], z[:-1, :])) #TODO: c2r?
        z0 = jnp.repeat(z, dims, axis=-1)
        z1 = jnp.tile(z, (1, dims))
        rt = jax.vmap(lambda a, b: a[0] * b.conj(), in_axes=-1, out_axes=0)(z0, z1).reshape(r.shape)
        r = beta * r + (1 - beta) * rt # exponential moving average
        rhat = r / (1 - betapow) # bias correction due to small beta
        r_sqsum = jnp.sum(jnp.abs(rhat)**2, axis=-1)

        v = mimo(w, u)
        lcma = jnp.sum(jnp.abs(jnp.abs(v)**2 - R2)**2)
        lmu = 2 * (jnp.sum(r_sqsum) - jnp.sum(jnp.diag(r_sqsum)))
        gcma = 4 * (v * (jnp.abs(v)**2 - R2))[..., None, None] * jnp.conj(u).T[None, ...]
        gmu_tmp_full = (4 * rhat[..., None, None]
                        * z.T[None, ..., None, None]
                        * jnp.conj(u).T[None, None, None, ...]) # shape: [dims, dims, delta, dims, T]
        # reduce delta axis first
        gmu_tmp_dr = jnp.sum(gmu_tmp_full, axis=2) # shape: [dims, dims, dims, T]
        # cross correlation = full correlation - self correlation
        gmu = jnp.sum(gmu_tmp_dr, axis=1) - gmu_tmp_dr[jnp.arange(dims), jnp.arange(dims), ...]
        l = lcma + lmu
        g = gcma + gmu

        out = (w, l)
        w = w - lr(i) * g
        betapow *= beta
        state = (w, z, r, betapow)
        return state, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def rde(
    lr: Union[float, Schedule] = 2**-15,
    train: Union[bool, Schedule] = False,
    Rs: Array = jnp.array([0.4472136 , 1., 1.34164079]),
    const: Optional[Array] = None
) -> AdaptiveFilter:
    """Radius Directed adaptive Equalizer

    Args:
      lr: learning rate. scalar or Schedule
      train: schedule training mode, which can be a bool for global control within one call
        or an array of bool to swich training on iteration basis
      Rs: the radii of the target constellation
      const: Optional; constellation used to infer R2 when R2 is None

    Returns:
      an ``AdaptiveFilter`` object

    References:
      - [1] Fatadin, I., Ives, D. and Savory, S.J., 2009. Blind equalization and
        carrier phase recovery in a 16-QAM optical coherent system. Journal
        of lightwave technology, 27(15), pp.3042-3049.
    """
    lr = make_schedule(lr)
    train = make_schedule(train)

    if const is not None:
        Rs = jnp.array(jnp.unique(jnp.abs(const)))
    else:
        Rs = jnp.array(Rs)

    def init(dims=2, w0=None, taps=32, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike)
        s0 = w0,
        return s0

    def loss_fn(w, u, x, i):
        v = r2c(mimo(w, u))[None,:]
        R2 = jnp.where(train(i),
                       jnp.abs(x)**2,
                       Rs[jnp.argmin(
                           jnp.abs(Rs[:,None] * v / jnp.abs(v) - v),
                           axis=0)]**2)
        l = jnp.sum(jnp.abs(R2 - jnp.abs(v[0,:])**2))
        return l

    def update(i, s, inp):
        w, = s
        u, x = inp
        l, g = jax.value_and_grad(loss_fn)(w, u, x, i)
        out = (w, l)
        w = w - lr(i) * g.conj()
        s = w,
        return s, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    update.train = train

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def rls_rde(
    λ: Union[float, Schedule] = 0.999,
    δ: float = 0.01,
    train: Union[bool, Schedule] = False,
    Rs: Array = jnp.array([0.4472136 , 1., 1.34164079]),
    const: Optional[Array]=None
) -> AdaptiveFilter:
    """RLS-RDE blind MIMO adaptive filter.
    λ: forgetting factor (λ<1)
    δ: is small positive number
    [1] Md. S. Faruk and S. J. Savory, “Digital Signal Processing for
     Coherent Transceivers Employing Multilevel Formats,” J. Lightwave
     Technol., vol. 35, no. 5, pp. 1125-1141, Mar. 2017, doi: 10.1109/JLT.2017.2662319.

    """
    _λ = make_schedule(λ)
    train = make_schedule(train)

    if const is not None:
        Rs = jnp.asarray(jnp.unique(jnp.abs(const)))
    else:
        Rs = jnp.asarray(Rs)

    def init(w0=None, taps=19, dims=2, dtype=default_complexing_dtype(), nspike=1):
        w0 = mimoinitializer(taps, dims, dtype, initkind='centralspike', nspike=nspike) if w0 is None else w0
        P0 = jnp.tile(δ * jnp.eye(taps * dims, dtype=dtype), (dims, 1, 1))
        return (w0, P0)

    @partial(jax.vmap, in_axes=(None, 0, (None, 0)), out_axes=(0, -1))
    def update(i, s, inp):
        w, _P = s
        u, x = inp
        N, dims = u.shape[0], w.shape[0]
        # in case of polyphase filtering, P is downsized
        i_P = jnp.ix_(jnp.arange(N*dims), jnp.arange(N*dims))
        P = _P[i_P]

        u_i = u.reshape(-1, order='F')[:, None]
        h = w.conj().reshape(-1)[:, None]
        λ = _λ(i)

        z = u_i @ u_i.conj().T @ h
        k = 1/λ * P @ z / (1 + 1/λ * z.conj().T @ P @ z)
        vs = jnp.sqrt(jnp.abs(jnp.squeeze(h.conj().T @ z)))
        R2 = jnp.where(train(i),
                       jnp.abs(x)**2,
                       Rs[jnp.argmin(jnp.abs(Rs - vs))]**2)
        ε = R2 - vs**2
        h = h + k * ε.conj()
        P = 1/λ * P - 1/λ * k * z.conj().T @ P

        w = h.conj().reshape((dims, -1))

        _P = _P.at[i_P].set(P)
        s = w, _P
        out = (ε,)
        return s, out

    def apply(s, y):
        w = tuple(s)[0]
        return mimo(w, y)

    update.train = train

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def frame_cpr_kf(Q: Array = jnp.array([[0,    0],
                                       [0, 1e-6]]), # 1e-8 is better if akf is False
                 R: Array = jnp.array([[1e-2, 0],
                                       [0, 1e-3]]),
                 const: Array = sym_map.const("16QAM", norm=True),
                 train: Union[bool, Schedule] = False,
                 akf: Union[bool, Schedule] = False,
                 alpha: float = 0.98) -> AdaptiveFilter:
    """Block based estimator of carrier frequency offset

    frame-by-frame coarse carrier phsae recovery using Kalman filter, can tolerate 0.1 * baudrate
    frequency offset[1].
    
    Args:
        Q: covariance matrix of observer noises
        R: covariance matrix of system noises
        const: reference constellation used in decison stage
        train: scheduler of training mode
        akf: scheduler of AKF
        alpha: smoothening factor used in AKF
    
    Returns:
        A ``AdaptiveFilter`` object
    
    Caution:
        needs proper initialization of FO[1]
    
    References:
        - [1] Inoue, Takashi, and Shu Namiki. "Carrier recovery for M-QAM signals based on
          a block estimation process with Kalman filter." Optics express 22.13 (2014): 15376-15387.
        - [2] Akhlaghi, Shahrokh, Ning Zhou, and Zhenyu Huang. "Adaptive adjustment of noise
          covariance in Kalman filter for dynamic state estimation." 2017 IEEE power & energy
          society general meeting. IEEE, 2017.
    """
    const = jnp.asarray(const)
    train = make_schedule(train)
    akf = make_schedule(akf)

    def init(w0=0):
        z0 = jnp.array([[0], [w0]], dtype=jnp.float32)
        P0 = jnp.zeros((2, 2), dtype=jnp.float32)
        state0 = (z0, P0, Q)
        return state0

    def update(i, state, inp):
        z_c, P_c, Q = state
        y, x = inp

        N = y.shape[0] # frame size
        A = jnp.array([[1, N],
                       [0, 1]])
        I = jnp.eye(2)
        n = (jnp.arange(N) - (N - 1) / 2)

        z_p = A @ z_c
        P_p = A @ P_c @ A.T + Q
        phi_p = z_p[0, 0] + n * z_p[1, 0]  # linear approx.
        s_p = y * jnp.exp(-1j * phi_p)
        d = jnp.where(train(i), x, const[jnp.argmin(jnp.abs(const[None, :] - s_p[:, None]), axis=-1)])
        scd_p = s_p * d.conj()
        sumscd_p = jnp.sum(scd_p)
        e = jnp.array([[jnp.arctan(sumscd_p.imag / sumscd_p.real)],
                       [(jnp.sum(n * scd_p)).imag / (jnp.sum(n * n * scd_p)).real]])

        G = P_p @ jnp.linalg.pinv((P_p + R))
        z_c = z_p + G @ e
        P_c = (I - G) @ P_p

        Q = jnp.where(akf(i),
                      alpha * Q + (1 - alpha) * (G @ e @ e.T @ G),
                      Q)

        out = (z_p[1, 0], phi_p)
        state = (z_c, P_c, Q)

        return state, out

    def apply(phis, ys):
        return jax.vmap(lambda y, phi: y * jnp.exp(-1j * phi))(ys, phis)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter)
def foe_YanW(lr: Union[float, Schedule] = 1e-6):
    '''
    [1] Wang, Yan, Erchin Serpedin, and Philippe Ciblat. "Optimal blind nonlinear 
    least-squares carrier phase and frequency offset estimation for general QAM 
    modulations." IEEE Transactions on wireless communications 2.5 (2003): 1040-1054.
    '''
    lr = make_schedule(lr)

    def F(x):
        y = jnp.piecewise(
            x,
            [x < 0.7236, x >= 1.1708],
            # [lambda x: 122.2733 * x, lambda x: 331.885 * x - 30.4524, 0.]
            [lambda x: 0.36 * x, lambda x: x - 0.09, 0.]
            )
        return y

    def init(w0=0.):
        s0 = jnp.asarray(w0*4),
        return s0

    def loss(w, x):
        N = x.shape[0]
        n = jnp.arange(N)
        y = F(jnp.abs(x)) * jnp.exp(4j*jnp.angle(x))
        l = -jnp.abs((y * jnp.exp(-1j*w*n)).sum())**2 / N
        return l

    def update(i, s, x):
        N = x.shape[0]
        w, = s
        l, g = jax.value_and_grad(loss)(w, x)
        out = (w / 4, l)
        w = w - lr(i) * g / (N**2) # norm grad by N^2
        s = w,
        return s, out

    def apply(s, y):
        w, = s
        N = y.shape[0]
        n = jnp.arange(N)
        return y * jnp.exp(-1j * w * n)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter)
def foe_YanW_ekf(
    q: float = 1e-15,           # process noise (constant)
    r: float = 1e-1,            # measurement noise (constant)
    p0: float = 1e-9,           # initial covariance (constant)
):
    '''
    EKF-based frequency offset estimation [1].
    Parameters are constant (no N-scaling) since the loss function
    already normalizes the gradient by N².

    Args:
        q: Process noise. Models frequency drift between blocks.
            Larger q → faster tracking, noisier estimates.

        r: Measurement noise. Models gradient estimation noise.
            Larger r → smoother estimates, slower response.

        p0: Initial covariance.
            Larger p0 → faster initial convergence.

    Note:
        Smaller block sizes (N) converge slower due to the fundamental
        O(1/N³) scaling of frequency estimation variance (Cramér-Rao bound).

    Reference:
    [1] Wang, Yan, Erchin Serpedin, and Philippe Ciblat. "Optimal blind nonlinear
    least-squares carrier phase and frequency offset estimation for general QAM
    modulations." IEEE Transactions on wireless communications 2.5 (2003): 1040-1054.
    '''

    def F(x):
        y = jnp.piecewise(
            x,
            [x < 0.7236, x >= 1.1708],
            [lambda x: 0.36 * x, lambda x: x - 0.09, 0.]
        )
        return y

    def init(w0=0.):
        w = jnp.asarray(w0 * 4)
        # Use -1 as sentinel; actual P computed on first update when N is known
        P = jnp.asarray(-1.0)
        return (w, P)

    def loss(w, x):
        N = x.shape[0]
        n = jnp.arange(N)
        y = F(jnp.abs(x)) * jnp.exp(4j * jnp.angle(x))
        # N² normalization for N-agnostic gradient
        l = -jnp.abs((y * jnp.exp(-1j * w * n)).sum())**2 / N**2
        return l

    def update(i, s, x):
        w, P = s

        # Initialize P on first iteration (sentinel = -1)
        P = jnp.where(P < 0, p0, P)

        # Predict
        P_pred = P + q

        # Gradient and Hessian via autodiff (innovation: d = g, expect g → 0)
        g = jax.grad(loss)(w, x)
        H = jax.grad(jax.grad(loss))(w, x)

        # Use |H| for numerical stability (H should be negative at maximum)
        H_abs = jnp.maximum(jnp.abs(H), 1e-12)

        # Kalman gain
        S = H_abs**2 * P_pred + r
        K = P_pred * H_abs / S

        # Output before update
        out = (w / 4, loss(w, x))

        # State update
        w_new = w - K * g
        P_new = (1 - K * H_abs) * P_pred
        P_new = jnp.maximum(P_new, 1e-16)  # prevent negative variance

        return (w_new, P_new), out

    def apply(s, y):
        w, P = s
        N = y.shape[0]
        n = jnp.arange(N)
        return y * jnp.exp(-1j * w * n)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def cpane_ekf(train: Union[bool, Schedule] = False,
              alpha: float = 0.99,
              beta: float = 0.6,
              Q: complex = 1e-4 + 0j,
              R: complex = 1e-2 + 0j,
              akf: bool = True,
              const: Array = sym_map.const("16QAM", norm=True)) -> AdaptiveFilter:
    """Carrier Phase and Amplitude Noise Estimator (single-dimensional).

    Symbol-by-symbol fine carrier phase recovery using extended Kalman filter.
    For multi-dimensional processing, use eqx.filter_vmap.

    Args:
      train: scheduler for training mode
      alpha: smoothing factor for Q/R adaptation
      beta: smoothing factor for phase averaging
      Q: process noise covariance
      R: measurement noise covariance
      akf: adaptive controlling of Q and R (AKF)
      const: reference constellation

    Returns:
      an ``AdaptiveFilter`` object

    Example:
        @eqx.filter_vmap
        def make_cpr(_):
            return eq.CPR(kernel=ak.cpane_ekf())
        cpr = make_cpr(jnp.arange(2))  # dual-pol

    References:
      - [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
        of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
      - [2] Akhlaghi, Shahrokh, Ning Zhou, and Zhenyu Huang. "Adaptive adjustment of noise
        covariance in Kalman filter for dynamic state estimation." 2017 IEEE power & energy
        society general meeting. IEEE, 2017.
    """
    const = jnp.asarray(const)
    train = make_schedule(train)

    def init(p0=0j):
        state0 = (jnp.asarray(p0), jnp.asarray(1j), jnp.asarray(0j),
                  jnp.asarray(Q), jnp.asarray(R))
        return state0

    def update(i, state, inp):
        Psi_c, P_c, Psi_a, _Q, _R = state
        y, x = inp

        Psi_p = Psi_c
        P_p = P_c + _Q
        # exponential moving average
        Psi_a = beta * Psi_a + (1 - beta) * Psi_c

        d = jnp.where(train(i),
                      x,
                      const[jnp.argmin(jnp.abs(const - y * jnp.exp(-1j * Psi_a)))])

        H = 1j * d * jnp.exp(1j * Psi_p)
        K = P_p * H.conj() / (H * P_p * H.conj() + _R)
        v = y - d * jnp.exp(1j * Psi_p)

        out = (Psi_c, (_Q, _R))

        Psi_c = Psi_p + K * v
        P_c = (1. - K * H) * P_p
        e = y - d * jnp.exp(1j * Psi_c)
        _Q = alpha * _Q + (1 - alpha) * K * v * v.conj() * K.conj() if akf else _Q
        _R = alpha * _R + (1 - alpha) * (e * e.conj() + H * P_p * H.conj()) if akf else _R

        state = (Psi_c, P_c, Psi_a, _Q, _R)

        return state, out

    def apply(s, y):
        Psi = tuple(s)[0]
        return y * jnp.exp(-1j * Psi)

    update.train = train

    return AdaptiveFilter(init, update, apply)


def cpr_4thpower_pll(
    mu: float = 0.01,
    M: int = 4,
) -> AdaptiveFilter:
    """4th-power PLL for carrier phase recovery.

    Removes QAM modulation via M-th power, then tracks phase with a PLL.
    Handles any initial phase offset without ambiguity during tracking.

    Single-dimensional: use with eq.CPR and eqx.filter_vmap for ensembles.

    Note: Has π/2 ambiguity in final output for M=4 (square QAM).
    Resolve with differential decoding or final block alignment if needed.

    Args:
        mu: Loop bandwidth / learning rate.
            Larger = faster tracking, more noise.
            Typical: 0.001-0.1
        M: Power to raise signal (4 for square QAM, 2 for QPSK).
            Removes modulation: symbol phases become multiples of 2π/M.

    Returns:
        AdaptiveFilter with single-dimensional state.

    Example:
        @eqx.filter_vmap
        def make_cpr_ensemble(_):
            return eq.CPR(kernel=ak.cpr_4thpower_pll(mu=0.01))
        cpr = make_cpr_ensemble(jnp.arange(2))
    """

    def init(phase0=0.0):
        return jnp.asarray(phase0)

    def update(i, phase, y):
        # M-th power removes QAM modulation
        # For 16-QAM: symbol phases are k*π/2, so y^4 has phase 4*carrier_phase
        y_M = y ** M

        # Expected phase after M-th power
        expected = M * phase

        # Phase error (unwrapped to [-π, π])
        err = jnp.angle(y_M) - expected
        err = jnp.arctan2(jnp.sin(err), jnp.cos(err))  # wrap to [-π, π]

        # Scale back to carrier phase domain
        err_scaled = err / M

        # PLL update
        phase_new = phase + mu * err_scaled

        # Keep phase in reasonable range to avoid numerical issues
        phase_new = jnp.mod(phase_new + jnp.pi, 2*jnp.pi) - jnp.pi

        return phase_new, err_scaled

    def apply(phase, y):
        return y * jnp.exp(-1j * phase)

    return AdaptiveFilter(init, update, apply)


def cpr_partition_pll(
    mu: float = 0.01,
    const: Array = sym_map.const('16QAM', norm=True),
) -> AdaptiveFilter:
    """Partition-based PLL for 16-QAM carrier phase recovery.

    Uses outer ring symbols (at ±45°, ±135°) for robust phase tracking.
    Works with arbitrary initial phase offset including 45°.

    The outer ring of normalized 16-QAM has amplitude ~1.34 and phases
    at ±45°, ±135°. By detecting these symbols and comparing to expected
    phases, we can track carrier phase without ambiguity.

    Args:
        mu: Loop bandwidth / learning rate.
            Smaller = slower but more stable.
            Typical: 0.001-0.05
        const: Reference constellation (default: normalized 16-QAM).
            Used to compute amplitude threshold automatically.

    Returns:
        AdaptiveFilter with single-dimensional state.

    Example:
        cpr = eq.CPR(kernel=ak.cpr_partition_pll(mu=0.01))
    """
    const = jnp.asarray(const)

    # Compute amplitude rings from constellation
    amps = jnp.abs(const)
    unique_amps = jnp.unique(jnp.round(amps, decimals=3))  # ~[0.447, 1.0, 1.342]
    # Threshold between middle and outer ring
    amp_threshold = (unique_amps[-2] + unique_amps[-1]) / 2  # ~1.17 for 16-QAM

    # Expected phases for outer ring of 16-QAM (±3, ±3) symbols
    # At 45°, 135°, -135°, -45°
    outer_phases = jnp.array([jnp.pi/4, 3*jnp.pi/4, -3*jnp.pi/4, -jnp.pi/4])

    def init(phase0=0.0):
        return jnp.asarray(phase0)

    def update(i, phase, y):
        # Rotate by current phase estimate
        y_rot = y * jnp.exp(-1j * phase)

        # Amplitude-based weight: outer ring contributes more
        # Soft weighting avoids hard decisions
        amp = jnp.abs(y)
        weight = jnp.clip((amp - amp_threshold) / 0.2, 0.0, 1.0)

        # Find phase of rotated symbol
        actual_phase = jnp.angle(y_rot)

        # Find nearest outer ring phase
        diffs = actual_phase - outer_phases
        diffs = jnp.arctan2(jnp.sin(diffs), jnp.cos(diffs))  # wrap to [-π, π]
        nearest_idx = jnp.argmin(jnp.abs(diffs))
        err = diffs[nearest_idx]

        # Weighted PLL update (outer ring symbols weighted more)
        phase_new = phase + mu * weight * err
        phase_new = jnp.mod(phase_new + jnp.pi, 2*jnp.pi) - jnp.pi

        return phase_new, err

    def apply(phase, y):
        return y * jnp.exp(-1j * phase)

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def anf(f0: float,
        sr: float,
        A: float = 1,
        phi: float = 0,
        lr: float = 1e-4) -> AdaptiveFilter:
    """ Adaptive Notch Filter for noise cancelling

    Args:
        f0: target frequency
        sr: sampling rate
        A: amplitude
        phi: phase
        lr: learning rate
    
    Returns:
        a ``AdaptiveFilter`` object

    References:
        - [1] Widrow, Bernard, et al. "Adaptive noise cancelling: Principles and applications."
          Proceedings of the IEEE 63.12 (1975): 1692-1716.
        - [2] Li, Fan, et al. "100 Gbit/s PAM4 signal transmission and reception for 2-km
          interconnect with adaptive notch filter for narrowband interference." Optics express
          26.18 (2018): 24066-24074.
    """
    lr = make_schedule(lr)
    T = 1 / sr
    ω0 = 2 * np.pi * f0

    def init(w0=None):
        w0 = jnp.array([0., 0.], dtype=default_complexing_dtype()) if w0 is None else w0
        state0 = w0,
        return state0

    def update(i, state, inp):
        w, = state
        d = inp
        x = jnp.array([A * np.cos(ω0 * i * T + phi), A * np.sin(ω0 * i * T + phi)])
        y = jnp.inner(w, x)
        e = d - y
        w += 2 * lr(i) * e * x
        state = w,
        return state, e

    def apply(es, ys):
        return ys - es

    return AdaptiveFilter(init, update, apply)



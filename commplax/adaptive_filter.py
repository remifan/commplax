import jax
import functools
import numpy as np
from typing import Any, Callable, NamedTuple, Tuple, Union
from functools import partial
from jax import lax, jit, numpy as jnp
from commplax import comm, xop
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node

Array = Any
Params = Any
State = Any   # internal State
State = Any   # internal State
Signal = Any # Gradient updates are of the same type as parameters
AFState = Any

Step = int
InitFn = Callable
UpdateFn = Callable[[AFState, Signal], AFState]
StaticMapFn = Callable[[Any, Any], Any]

cpus = jax.devices("cpu")
gpus = jax.devices("gpu")


class AdaptiveFilter(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    eval_fn: StaticMapFn


def adaptive_filter(af_maker, trainable=False):
    @functools.wraps(af_maker)
    def _af_maker(*args, **kwargs):
        init, update, static_map = af_maker(*args, **kwargs)

        @functools.wraps(init)
        def _init(*args, **kwargs):
            x0 = init(*args, **kwargs)
            return jax.device_put(x0)

        @jax.jit
        @functools.wraps(update)
        def _update(af_state, af_inp):
            if trainable:
                af_inp = af_inp if isinstance(af_inp, tuple) else (af_inp,)
                af_inp = (af_inp + (0., False))[:3]
            af_inp = jax.device_put(af_inp)
            af_state, af_out = update(af_state, af_inp)
            return af_state, af_out

        @jax.jit
        @functools.wraps(static_map)
        def _static_map(af_ps, af_xs):
            return static_map(af_ps, af_xs)

        return AdaptiveFilter(_init, _update, _static_map)

    return _af_maker


def array(af_maker, replicas, axis=-1):
    @functools.wraps(af_maker)
    def rep_af_maker(*args, **kwargs):
        init, update, static_map = af_maker(*args, **kwargs)

        @functools.wraps(init)
        def rep_init(*args, **kwargs):
            x0 = init(*args, **kwargs)
            x0_flat, x0_tree = tree_flatten(x0)
            x0_flat = tuple(map(lambda v: jnp.repeat(v[..., None], replicas, axis=axis), x0_flat))
            x0 = tree_unflatten(x0_tree, x0_flat)
            return x0

        @jax.jit
        @functools.wraps(update)
        def rep_update(af_state, af_inp):
            af_state, af_out = jax.vmap(update, in_axes=axis, out_axes=axis)(af_state, af_inp)
            return af_state, af_out

        @jax.jit
        @functools.wraps(static_map)
        def rep_static_map(af_ps, af_xs):
            return jax.vmap(static_map, in_axes=axis, out_axes=axis)(af_ps, af_xs)

        return AdaptiveFilter(rep_init, rep_update, rep_static_map)

    return rep_af_maker


def mimo(w, u):
    return jnp.einsum('ijt,tj->i', w, u)


def unitarize_mimo_weights(w):
    if len(w.shape) != 3 or w.shape[0] != w.shape[1]:
        raise ValueError('bad shaped weights, must be like (a, a, b)')
    if w.shape[0] != 2:
        raise ValueError('unitarization is not yet applicable to MIMO with shape other than 2x2')
    w = np.array(w)
    w[1, 0] = -w[0, 1][::-1].conj()
    w[1, 1] = w[0, 0][::-1].conj()
    return w


def make_train_argin(ys, truth=None, train=None):
    leny = ys.shape[0]
    if truth is not None:
        lent = truth.shape[0]
        if lent > leny:
            truth = truth[:leny]
        if train is None:
            T = jnp.full(lent, True)
            F = jnp.full(leny - lent, False)
            train = jnp.concatenate([T, F])
        elif train is True:
            train = jnp.full(leny, True)
        elif train is False:
            train = jnp.full(leny, False)
            truth = jnp.zeros_like(ys)
        elif isinstance(train, float):
            train = int(train * leny)
            T = jnp.full(train, True)
            F = jnp.full(leny - train, False)
            train = jnp.concatenate([T, F])
        elif isinstance(train, int):
            T = jnp.full(train, True)
            F = jnp.full(leny - train, False)
            train = jnp.concatenate([T, F])
        else:
            pass
        truth = jnp.concatenate([truth, jnp.zeros_like(ys, shape=(leny - lent,) + ys.shape[1:])])
        xs = (ys, truth, train)
    else:
        xs = ys
    return jax.device_put(xs)


def iterate(update, state, signal, truth=None, train=None, device=cpus[0]):
    xs = make_train_argin(signal, truth, train)
    return xop.scan(update, state, xs, jit_device=device)


@adaptive_filter
def cma(lr=1e-4, R2=1.32):
    def init(w0=None, taps=19, dims=2, unitarize=True):
        if w0 is None:
            w0 = np.zeros((2, 2, taps), dtype=np.complex64)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.
        elif unitarize:
            try:
                w0 = unitarize_mimo_weights(w0)
            except:
                pass
        return w0

    def update(w, u):
        loss_fn = lambda w, u: jnp.sum(jnp.abs(R2 - jnp.abs(mimo(w, u))**2))
        l, g = jax.value_and_grad(loss_fn)(w, u)
        o = (l, w)
        w = w - lr * g.conj()
        return w, o

    def static_map(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, static_map)


@partial(adaptive_filter, trainable=True)
def rde(lr=1e-4, Rs=jnp.unique(jnp.abs(comm.const("16QAM", norm=True)))):
    '''
    References:
    [1] Fatadin, I., Ives, D. and Savory, S.J., 2009. Blind equalization and
        carrier phase recovery in a 16-QAM optical coherent system. Journal
        of lightwave technology, 27(15), pp.3042-3049.
    '''
    def init(w0=None, taps=19, dims=2, unitarize=False):
        if w0 is None:
            w0 = np.zeros((2, 2, taps), dtype=np.complex64)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.
        elif unitarize:
            try:
                w0 = unitarize_mimo_weights(w0)
            except:
                pass
        return w0

    def update(w, inp):
        u, Rx, train = inp

        def loss_fn(w, u):
            v = mimo(w, u)[None,:]
            R2 = jnp.where(train,
                           Rx**2,
                           Rs[jnp.argmin(
                               jnp.abs(Rs[:,None] * v / jnp.abs(v) - v),
                               axis=0)]**2)
            l = jnp.sum(jnp.abs(R2 - jnp.abs(v[0,:])**2))
            return l

        l, g = jax.value_and_grad(loss_fn)(w, u)
        out = (l, w)
        w = w - lr * g.conj()
        return w, out

    def static_map(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, static_map)


@partial(adaptive_filter, trainable=True)
def ddlms(mu_w=1/2**10, mu_f=1/2**6, mu_s=0., grad_max=(50., 50.), eps=1e-8,
          const=comm.const("16QAM", norm=True)):
    '''
    Impl. follows Fig. 6 in [1]
    References:
    [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of
        finite-impulse-response filters tolerant to carrier-phase fluctuations
        in digital coherent optical receivers for higher-order quadrature
        amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    '''
    def init(state0):
        return state0

    def update(state, inp):
        w, f, s, = state
        u, x, train = inp

        v = mimo(w, u)

        z = v * f * s

        d = jnp.where(train, x, const[jnp.argmin(jnp.abs(const[:,None] - z[None,:]), axis=0)])

        psi_hat = jnp.abs(f)/f * jnp.abs(s)/s
        e_p = d * psi_hat - v
        e_f = d - f * v
        e_s = d - s * f * v
        gs = -1. / (jnp.abs(f * v)**2 + eps) * e_s * (f * v).conj()
        gf = -1. / (jnp.abs(v)**2 + eps) * e_f * v.conj()

        # clip the grads of f and s which are less regulated than w,
        # it may stablize this algo. in some corner cases?
        gw = -e_p[:, None, None] * u.conj().T[None, ...]
        gf = jnp.where(jnp.abs(gf) > grad_max[0], gf / jnp.abs(gf) * grad_max[0], gf)
        gs = jnp.where(jnp.abs(gs) > grad_max[1], gs / jnp.abs(gs) * grad_max[1], gs)

        out = (w, f, s, d)

        # update
        w = w - mu_w * gw
        f = f - mu_f * gf
        s = s - mu_s * gs

        state = (w, f, s)

        return state, out

    def static_map(ps, yf):
        ws, fs, ss = ps
        return jax.vmap(mimo)(ws, yf) * fs * ss

    return AdaptiveFilter(init, update, static_map)


@partial(adaptive_filter, trainable=True)
def cpane_ekf(beta=0.8,
              Q=1e-5 * (1.+1j),
              R=1e-2 * (1.+1j),
              const=comm.const("16QAM", norm=True)):
    '''
    References:
    [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
    of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
    '''
    const = jnp.array(const)

    def init(p0=0j):
        state0 = (p0, 1j, 0j)
        return state0

    def update(state, inp):

        Psi_c, P_c, Psi_a = state
        y, x, train = inp

        Psi_p = Psi_c
        P_p = P_c + Q

        Psi_a = beta * Psi_a + (1 - beta) * Psi_c

        d = jnp.where(train,
                      x,
                      const[jnp.argmin(jnp.abs(const - y * jnp.exp(-1j * Psi_a)))])

        H = 1j * d * jnp.exp(1j * Psi_p)
        K = P_p * H.conj() / (H * P_p * H.conj() + R)
        v = y - d * jnp.exp(1j * Psi_p)

        out = (Psi_c, d)

        Psi_c = Psi_p + K * v
        P_c = (1. - K * H) * P_p

        state = (Psi_c, P_c, Psi_a)

        return state, out

    def static_map(Psi, ys):
        return ys * jnp.exp(-1j * Psi)

    return AdaptiveFilter(init, update, static_map)


@adaptive_filter
def anf(f0, sr, A=1, phi=0, mu=1e-4):
    '''
    References:
    [1] Widrow, Bernard, et al. "Adaptive noise cancelling: Principles and applications."
        Proceedings of the IEEE 63.12 (1975): 1692-1716.
    [2] Li, Fan, et al. "100 Gbit/s PAM4 signal transmission and reception for 2-km
        interconnect with adaptive notch filter for narrowband interference." Optics express
        26.18 (2018): 24066-24074.
    '''
    T = 1 / sr
    w0 = 2 * np.pi * f0

    def init(w0=jnp.array([0., 0.])):
        state0 = (w0, 0)
        return state0

    def update(state, inp):
        w, i = state
        d = inp
        x = jnp.array([A * np.cos(w0 * i * T + phi), A * np.sin(w0 * i * T + phi)])
        y = jnp.inner(w, x)
        e = d - y
        w += 2 * mu * e * x
        i += 1
        state = (w, i)
        return state, e

    def static_map(es, ys):
        return ys - es

    return AdaptiveFilter(init, update, static_map)


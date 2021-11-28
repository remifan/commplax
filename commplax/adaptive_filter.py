# Copyright 2021 The Commplax Authors.
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


import jax
import functools
import numpy as np
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
from functools import partial
from jax import numpy as jnp
from commplax import comm, xcomm, xop, cxopt
from jax.tree_util import tree_flatten, tree_unflatten
from jax.lax import stop_gradient
from commplax.cxopt import Schedule

Array = Any
Params = Any
State = Any   # internal State
Signal = Any # Gradient updates are of the same type as parameters
AFState = Any

Step = int
InitFn = Callable
UpdateFn = Callable[[Step, AFState, Signal], AFState]
ApplyFn = Callable[[Any, Any], Any]


class AdaptiveFilter(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    eval_fn: ApplyFn


def adaptive_filter(af_maker: Callable, trainable=False):
    @functools.wraps(af_maker)
    def _af_maker(*args, **kwargs):
        init, update, apply = af_maker(*args, **kwargs)

        @functools.wraps(init)
        def _init(*args, **kwargs):
            x0 = init(*args, **kwargs)
            return jax.device_put(x0)

        @jax.jit
        @functools.wraps(update)
        def _update(i, af_state, af_inp):
            if trainable:
                af_inp = af_inp if isinstance(af_inp, tuple) else (af_inp,)
                af_inp = (af_inp + (0.,))[:2]
            else:
                af_inp = af_inp[0] if isinstance(af_inp, tuple) else af_inp
            af_inp = jax.device_put(af_inp)
            af_state, af_out = stop_gradient(update(i, af_state, af_inp))
            return af_state, af_out

        @jax.jit
        @functools.wraps(apply)
        def _apply(af_ps, af_xs):
            return apply(af_ps, af_xs)

        _update.trainable = trainable

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

        @jax.jit
        @functools.wraps(update)
        def rep_update(i, af_state, af_inp):
            af_state, af_out = jax.vmap(update, in_axes=(None, axis, axis), out_axes=axis)(i, af_state, af_inp)
            return af_state, af_out

        @jax.jit
        @functools.wraps(apply)
        def rep_apply(af_ps, af_xs):
            return jax.vmap(apply, in_axes=axis, out_axes=axis)(af_ps, af_xs)

        return AdaptiveFilter(rep_init, rep_update, rep_apply)

    return rep_af_maker


def frame(y, taps, sps, rtap=None):
    y_pad = jnp.pad(y, mimozerodelaypads(taps=taps, sps=sps, rtap=rtap))
    yf = jnp.array(xop.frame(y_pad, taps, sps))
    return yf


def iterate(update: UpdateFn,
            step0: Step,
            state: AFState,
            signal: Signal,
            truth=None,
            truth_ndim=2,
            device=None):
    steps = step0 + jnp.arange(signal.shape[0])
    # pad dummy truth
    truth = jnp.zeros((0, *signal.shape[1 - truth_ndim:]), dtype=signal.dtype) if truth is None else truth[:signal.shape[0]]
    padw_data_axes = ((0, 0),) * (truth_ndim - 1)
    truth = jnp.pad(truth, ((0, signal.shape[0] - truth.shape[0]), *padw_data_axes))
    xs = (steps, signal, truth)
    return steps[-1], xop.scan(lambda c, xs: update(xs[0], c, xs[1:]), state, xs, jit_device=device)


def mimo(w, u):
    return jnp.einsum('ijt,tj->i', w, u)


def r2c(r):
    ''' Pack real-valued signal into complex-valued signal
    for example, converting
    [[ 0.  0.  1. -1.]
     [ 2. -2.  3. -3.]
     [ 4. -4.  5. -5.]
     [ 6. -6.  7. -7.]]
    to
    [[0.+0.j 1.-1.j]
     [2.-2.j 3.-3.j]
     [4.-4.j 5.-5.j]
     [6.-6.j 7.-7.j]]
    '''
    if not jnp.iscomplexobj(r):
        if r.ndim != 2:
            raise ValueError('invalid ndim, expected 2 but got %d' % r.ndim)
        r = r.reshape((r.shape[0], r.shape[-1] // 2, -1))
        c = r[..., 0] + 1j * r[..., 1]
    else:
        c = r
    return c


def c2r(c):
    ''' Unpack complex-valued signal into real-valued signal
    for example, converting
    [[0.+0.j 1.-1.j]
     [2.-2.j 3.-3.j]
     [4.-4.j 5.-5.j]
     [6.-6.j 7.-7.j]]
    to
    [[ 0.  0.  1. -1.]
     [ 2. -2.  3. -3.]
     [ 4. -4.  5. -5.]
     [ 6. -6.  7. -7.]]
    '''
    if jnp.iscomplexobj(c):
        if c.ndim != 2:
            raise ValueError('invalid ndim, expected 2 but got %d' % c.ndim)
        r = jnp.stack([c.real, c.imag], axis=-1).reshape((c.shape[0], -1))
    else:
        r = c
    return r


def filterzerodelaypads(taps, stride=1, rtap=None):
    if rtap is None:
        rtap = (taps + 1) // 2 - 1
    filterdelay = int(np.ceil((rtap + 1) / stride) - 1)
    pads = np.array([[filterdelay * stride, taps - stride * (filterdelay + 1)], [0,0]])
    return pads


def mimozerodelaypads(taps, sps=2, rtap=None):
    return filterzerodelaypads(taps, sps, rtap)


def mimoinitializer(taps, dims, dtype, initkind):
    initkind = initkind.lower()
    if initkind == 'zeros':
        w0 = jnp.zeros((dims, dims, taps), dtype=dtype)
    elif initkind == 'centralspike':
        w0 = np.zeros((dims, dims, taps), dtype=dtype)
        ctap = (taps + 1) // 2 - 1
        w0[np.arange(dims), np.arange(dims), ctap] = 1.
        w0 = jnp.array(w0)
    else:
        raise ValueError('invalid initkind %s' % initkind)
    return w0


def decision(const, v, stopgrad=True):
    """ simple symbol decision based on Euclidean distance
    """
    if v.ndim > 1:
        raise ValueError(f'ndim = 1 is expected, but got {v.ndim} instead')

    dims = v.shape[-1]
    dim_axis = tuple(range(1, dims + 1))
    d = const[jnp.argmin(jnp.abs(jnp.expand_dims(const, axis=dim_axis) - v[None, ...]),
                         axis=0)]
    return stop_gradient(d) if stopgrad else d


@adaptive_filter
def lms(
    lr: Union[float, Schedule] = 1e-4,
    const: Optional[Array]=comm.const('16QAM', norm=True)
) -> AdaptiveFilter:
    """LMS MIMO adaptive filter.

    Args:
      lr: Optional; learning rate
      const: Optional; constellation used to infer R2 when R2 is None

    Returns:
      an ``AdaptiveFilter`` object
    """
    lr = cxopt.make_schedule(lr)
    if const is not None:
        const = jnp.asarray(const)

    def init(w0=None, taps=19, dims=2, dtype=np.complex64):
        if w0 is None:
            w0 = np.zeros((dims, dims, taps), dtype=dtype)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.
        return w0.astype(dtype)

    def loss_fn(w, u):
        v = r2c(mimo(w, u)[None, :])[0, :]
        d = decision(const, v)
        loss = jnp.sum(jnp.abs(d - v)**2)
        return loss

    def update(i, w, u):
        l, g = jax.value_and_grad(loss_fn)(w, u)
        out = (w, l)
        w = w - lr(i) * g.conj()
        return w, out

    def apply(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def cma(
    lr: Union[float, Schedule] = 1e-4,
    R2: Optional[float]=1.32,
    const: Optional[Array]=None
) -> AdaptiveFilter:
    """CMA blind MIMO adaptive filter.

    Args:
      lr: Optional; learning rate
      R2: Optional; reference square of radius
      const: Optional; constellation used to infer R2 when R2 is None

    Returns:
      an ``AdaptiveFilter`` object

    Examples:
      >>> from commplax.adaptive_filter import cma
      >>>
      >>> af = cma(lr=1e-4, R2=1.32) # for non-PS power normalized 16QAM signal

    References:
      - [1] D. Godard, “Self-recovering equalization and carrier tracking in two-dimensional data
        communication systems,” IEEE Trans. Commun., vol. 28, no. 11, pp. 1867–1875, Nov. 1980.
      - [2] K. Kikuchi, “Polarization-demultiplexing algorithm in the digital coherent receiver,”
        in Proc. Digest 2008 IEEE/LEOS Summer Topical Meetings, Jul., pp. 101–102.
    """
    lr = cxopt.make_schedule(lr)

    if const is not None:
        R2 = jnp.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))

    def init(w0=None, taps=19, dims=2, dtype=np.complex64):
        if w0 is None:
            w0 = np.zeros((dims, dims, taps), dtype=dtype)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.
        return w0.astype(dtype)

    def loss_fn(w, u):
        v = r2c(mimo(w, u)[None, :])[0, :]
        loss = jnp.sum(jnp.abs(R2 - jnp.abs(v)**2))
        return loss

    def update(i, w, u):
        l, g = jax.value_and_grad(loss_fn)(w, u)
        out = (w, l)
        w = w - lr(i) * g.conj()
        return w, out

    def apply(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, apply)


@adaptive_filter
def mucma(
    dims: int = 2,
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
    lr = cxopt.make_schedule(lr)

    if const is not None:
        R2 = jnp.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))

    def init(w0=None, taps=19, dtype=np.complex64):
        if w0 is None:
            w0 = np.zeros((dims, dims, taps), dtype=dtype)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.

        w0 = jnp.asarray(w0).astype(dtype)
        z = jnp.zeros((delta, dims), dtype=dtype)
        r = jnp.zeros((dims, dims, delta), dtype=dtype)
        return w0, z, r, jnp.asarray(beta)

    def update(i, state, u):
        w, z, r, betapow = state
        z = jnp.concatenate((r2c(mimo(w, u)[None, :]), z[:-1,:]))
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

    def apply(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def rde(
    lr: Union[float, Schedule] = 2**-15,
    train: Union[bool, Schedule] = False,
    Rs: Array = jnp.unique(jnp.abs(comm.const("16QAM", norm=True))),
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
    lr = cxopt.make_schedule(lr)
    train = cxopt.make_schedule(train)

    if const is not None:
        Rs = jnp.array(jnp.unique(jnp.abs(const)))

    def init(dims=2, w0=None, taps=32, dtype=np.complex64):
        if w0 is None:
            w0 = np.zeros((dims, dims, taps), dtype=dtype)
            ctap = (taps + 1) // 2 - 1
            w0[np.arange(dims), np.arange(dims), ctap] = 1.
        return w0

    def loss_fn(w, u, x, i):
        v = r2c(mimo(w, u)[None,:])
        R2 = jnp.where(train(i),
                       jnp.abs(x)**2,
                       Rs[jnp.argmin(
                           jnp.abs(Rs[:,None] * v / jnp.abs(v) - v),
                           axis=0)]**2)
        l = jnp.sum(jnp.abs(R2 - jnp.abs(v[0,:])**2))
        return l

    def update(i, w, inp):
        u, x = inp
        l, g = jax.value_and_grad(loss_fn)(w, u, x, i)
        out = (w, l)
        w = w - lr(i) * g.conj()
        return w, out

    def apply(ws, yf):
        return jax.vmap(mimo)(ws, yf)

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def ddlms(
    lr_w: Union[float, Schedule] = 1/2**6,
    lr_f: Union[float, Schedule] = 1/2**7,
    lr_s: Union[float, Schedule] = 0.,
    lr_b: Union[float, Schedule] = 1/2**11,
    train: Union[bool, Schedule] = False,
    grad_max: Tuple[float, float] = (30., 30.),
    eps: float = 1e-8,
    beta: float = 0.,
    const: Array = comm.const("16QAM", norm=True)
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
    lr_w = cxopt.make_schedule(lr_w)
    lr_f = cxopt.make_schedule(lr_f)
    lr_s = cxopt.make_schedule(lr_s)
    lr_b = cxopt.make_schedule(lr_b)
    train = cxopt.make_schedule(train)

    def init(taps=31, dims=2, dtype=jnp.complex64, mimoinit='zeros'):
        w0 = mimoinitializer(taps, dims, dtype, mimoinit)
        f0 = jnp.full((dims,), 1., dtype=dtype)
        s0 = jnp.full((dims,), 1., dtype=dtype)
        b0 = jnp.full((dims,), 0., dtype=dtype)
        fshat0 = jnp.full((dims,), 1., dtype=dtype)
        return (w0, f0, s0, b0, fshat0)

    def update(i, state, inp):
        w, f, s, b, fshat = state
        u, x = inp

        v = mimo(w, u)
        # v = r2c(mimo(w, u)[None, :])[0, :]
        k = v * f
        c = k * s
        z = c + b
        q = v * fshat + b
        d = jnp.where(train(i), x, decision(const, q))
        l = jnp.sum(jnp.abs(z - d)**2)

        psi_hat = jnp.abs(f)/f * jnp.abs(s)/s
        e_w = (d - b) * psi_hat - v
        e_f = d - b - k
        e_s = d - b - c
        e_b = d - z
        gw = -1. / ((jnp.abs(u)**2).sum() + eps) * e_w[:, None, None] * u.conj().T[None, ...]
        # gw = -e_w[:, None, None] * u.conj().T[None, ...]
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

    def apply(ps, yf):
        ws, fs, ss, bs = ps
        return jax.vmap(mimo)(ws, yf) * fs * ss + bs

    return AdaptiveFilter(init, update, apply)


@partial(adaptive_filter, trainable=True)
def frame_cpr_kf(Q: Array = jnp.array([[0,    0],
                                       [0, 1e-9]]), # 1e-8 is better if akf is False
                 R: Array = jnp.array([[1e-2, 0],
                                       [0, 1e-3]]),
                 const: Array = comm.const("16QAM", norm=True),
                 train: Union[bool, Schedule] = False,
                 akf: Schedule = cxopt.piecewise_constant([10, 500], [False, True, False]),
                 alpha: float = 0.999) -> AdaptiveFilter:
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
    train = cxopt.make_schedule(train)
    akf = cxopt.make_schedule(akf)

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


@partial(adaptive_filter, trainable=True)
def cpane_ekf(train: Union[bool, Schedule] = False,
              alpha: float = 0.99,
              beta: float = 0.6,
              Q: complex = 1e-4 + 0j,
              R: complex =1e-2 + 0j,
              akf: bool = True,
              const: Array = comm.const("16QAM", norm=True)) -> AdaptiveFilter:
    """Carrier Phase and Amplitude Noise Estimator
    symbol-by-symbol fine carrier phsae recovery using extended Kalman filter

    Args:
      train: scheduler for training mode
      alpha: smoothening factor
      beta: smoothening factor
      Q: covariance matrix of observer noises
      R: covariance matrix of system noises
      akf: adaptive controlling of Q and R, a.k.a, AKF
      const: reference constellation

    Returns:
      a ``AdaptiveFilter`` object

    References:
      - [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
        of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
      - [2] Akhlaghi, Shahrokh, Ning Zhou, and Zhenyu Huang. "Adaptive adjustment of noise
        covariance in Kalman filter for dynamic state estimation." 2017 IEEE power & energy
        society general meeting. IEEE, 2017.
    """
    const = jnp.asarray(const)
    train = cxopt.make_schedule(train)

    def init(p0=0j):
        state0 = (p0, 1j, 0j, Q, R)
        return state0

    def update(i, state, inp):
        Psi_c, P_c, Psi_a, Q, R = state
        y, x = inp

        Psi_p = Psi_c
        P_p = P_c + Q
        # exponential moving average
        Psi_a = beta * Psi_a + (1 - beta) * Psi_c

        d = jnp.where(train(i),
                      x,
                      const[jnp.argmin(jnp.abs(const - y * jnp.exp(-1j * Psi_a)))])

        H = 1j * d * jnp.exp(1j * Psi_p)
        K = P_p * H.conj() / (H * P_p * H.conj() + R)
        v = y - d * jnp.exp(1j * Psi_p)

        out = (Psi_c, (Q, R))

        Psi_c = Psi_p + K * v
        P_c = (1. - K * H) * P_p
        e = y - d * jnp.exp(1j * Psi_c)
        Q = alpha * Q + (1 - alpha) * K * v * v.conj() * K.conj() if akf else Q
        R = alpha * R + (1 - alpha) * (e * e.conj() + H * P_p * H.conj()) if akf else R

        state = (Psi_c, P_c, Psi_a, Q, R)

        return state, out

    def apply(Psi, ys):
        return ys * jnp.exp(-1j * Psi)

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
    lr = cxopt.make_schedule(lr)
    T = 1 / sr
    w0 = 2 * np.pi * f0

    def init(w0=jnp.array([0., 0.])):
        state0 = (w0, 0)
        return state0

    def update(i, state, inp):
        w, i = state
        d = inp
        x = jnp.array([A * np.cos(w0 * i * T + phi), A * np.sin(w0 * i * T + phi)])
        y = jnp.inner(w, x)
        e = d - y
        w += 2 * lr(i) * e * x
        i += 1
        state = (w, i)
        return state, e

    def apply(es, ys):
        return ys - es

    return AdaptiveFilter(init, update, apply)



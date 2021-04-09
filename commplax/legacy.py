import numpy as np
from jax import grad, value_and_grad, lax, jit, vmap, numpy as jnp, devices, device_put
from jax.ops import index, index_add, index_update
from functools import partial
from commplax import xop, comm

cpus = devices("cpu")
gpus = devices("gpu")

def update(state, inp):
    w, f, s, b, i, pi, po = state
    u, x, train = inp

    v = mimo(w, u)

    z = v * f * s + b
    d = jnp.where(train, x, const[jnp.argmin(jnp.abs(const[:,None] - z[None,:]), axis=0)])
    l = jnp.sum(jnp.abs(z - d)**2)

    psi_hat = jnp.abs(f)/f * jnp.abs(s)/s
    e_w = (d - b) * psi_hat - v
    e_f = d - b - f * v
    e_s = d - b - s * f * v
    gs = -1. / (jnp.abs(f * v)**2 + eps) * e_s * (f * v).conj()
    gf = -1. / (jnp.abs(v)**2 + eps) * e_f * v.conj()
    gb = z - d

    # clip the grads of f and s which are less regulated than w,
    # it may stablize this algo. in some corner cases?
    gw = -e_w[:, None, None] * u.conj().T[None, ...]
    gf = jnp.where(jnp.abs(gf) > grad_max[0], gf / jnp.abs(gf) * grad_max[0], gf)
    gs = jnp.where(jnp.abs(gs) > grad_max[1], gs / jnp.abs(gs) * grad_max[1], gs)

    pihat = pi / (1 - beta**(i + 1)) + 1e-8 # bias correction
    pohat = po / (1 - beta**(i + 1)) + 1e-8 # bias correction
    gain = jnp.sqrt(pohat / pihat)
    out = ((w, f, s, b,), (l, d, gain))

    # update
    w = w - mu_w * gw
    f = f - mu_f * gf
    s = s - mu_s * gs
    b = b - mu_b * gb
    pi = ((1 - beta) * jnp.mean(jnp.abs(u)**2, axis=0).sum() + beta * pi)
    po = ((1 - beta) * jnp.sum(jnp.abs(z)**2) + beta * po)
    i += 1

    # w *= (jnp.abs(f) * jnp.abs(s))[:, None, None]
    # f /= jnp.abs(f)
    # s /= jnp.abs(s)
    lb = gainboundaries[0] * (1. - 1. / (gamma * i + 1.))
    ub = gainboundaries[1] * (1. + 1. / (gamma * i))
    gainval = jnp.array([gainboundaries[0], 1., gainboundaries[1]])
    w /= gainval[jnp.sum(gain > jnp.array([lb, ub]))]

    state = (w, f, s, b, i, pi, po)


def mimo(w, u):
    v = jnp.einsum('ijt,tj->i', w, u) # no dimension axis
    return v


def dd_lms(signal, w_init, f_init=None, s_init=None, data=None, train=None, lr_w=1/2**10, lr_f=1/2**6, lr_s=0.,
           grad_max=(50., 50.), const=comm.const("16QAM", norm=True), device=cpus[0]):
    '''
    Impl. follows Fig. 6 in [1]
    References:
    [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of finite-impulse-response
    filters tolerant to carrier-phase fluctuations in digital coherent optical receivers for
    higher-order quadrature amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    '''

    if train is None:
        if data is None:
            data = np.full((signal.shape[0], signal.shape[-1]), 0, dtype=signal.dtype)
            train = np.full((signal.shape[0],), False)
        else:
            train = np.concatenate([np.full((data.shape[0],), True),
                                    np.full((signal.shape[0] - data.shape[0],), False)])
    else:
        if train.shape[0] != signal.shape[0] or data.shape[0] != signal.shape[0]:
           raise ValueError('invalid shape')

    dims = signal.shape[-1]
    if f_init is None:
        f_init = np.full((dims,), 1+0j, dtype=signal.dtype) # dummy initial value
    if s_init is None:
        s_init = np.full((dims,), 1+0j, dtype=signal.dtype) # dummy initial value

    params = (w_init, f_init, s_init, lr_w, lr_f, lr_s, grad_max, 1e-8, const)
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    ret = _dd_lms(params, inputs)

    return ret


@jit
def _dd_lms(params, inputs):
    _, ret = lax.scan(step_dd_lms, params, inputs)

    return ret


def step_dd_lms(params, inputs):
    w, f, s, mu_p, mu_f, mu_s, grad_max, eps, const = params
    u, x, train = inputs

    v = mimo(w, u)

    z = v * f * s

    d = lax.cond(
        train,
        None,
        lambda _: x,
        None,
        lambda _: const[jnp.argmin(jnp.abs(const[:,None] - z[None,:]), axis=0)]
    )

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

    outputs = (w, f, s, d)

    # update
    w = w - mu_p * gw
    f = f - mu_f * gf
    s = s - mu_s * gs

    params = (w, f, s, mu_p, mu_f, mu_s, grad_max, eps, const)

    return params, outputs


def cma(y_f, w_init, lr=1e-4, R2=1.32, device=cpus[0]):

    #const = comm.const("16QAM", norm=True)
    #R2 = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    params = (w_init, R2, lr)
    inputs = (y_f,)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_cma, params, inputs)

    return ret


@jit
def step_cma(params, inputs):
    w, R2, lr = params
    u, = inputs

    l, g = value_and_grad(loss_cma)(w, u, R2)

    outputs = (l, w)

    w = w - lr * g.conj()

    params = (w, R2, lr)

    return params, outputs


def loss_cma(w, u, R2):
    v = mimo(w, u)
    l = jnp.sum(jnp.abs(R2 - jnp.abs(v)**2))
    return l


def rls_cma(y_f, w_init, beta=0.999, delta=1e-4, const=comm.const("16QAM", norm=True), device=cpus[0]):
    '''
    References:
    [1] Faruk, M.S. and Savory, S.J., 2017. Digital signal processing for coherent
    transceivers employing multilevel formats. Journal of Lightwave Technology, 35(5), pp.1125-1141.
    '''

    R2 = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    N = y_f.shape[0]
    taps = w_init.shape[-1]
    dims = w_init.shape[0]
    # w_init: DxDxT -> DTxD
    w_init = jnp.reshape(w_init.conj(), (dims, dims * taps)).T
    cI = jnp.eye(dims * taps, dtype=y_f.dtype)
    P_init = delta * jnp.tile(cI[...,None], (1, 1, dims))
    y_f = jnp.reshape(y_f, (N, -1), order='F')

    params = (w_init, P_init, R2, beta)
    inputs = (y_f,)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_rls_cma, params, inputs)

    l, w = ret

    # w: NxDTxD -> NxDxDT
    w = jnp.moveaxis(w, 0, -1).T
    # w: NxDxDT -> NxDxDxT
    w = jnp.reshape(w, (N, dims, dims, taps)).conj()

    return l, w


@jit
def step_rls_cma(params, inputs):
    h, P, R2, beta = params
    u, = inputs
    bi = 1. / beta

    def f(u, h, P):
        u = u[:,None]
        h = h[:,None]

        z = u @ u.T.conj() @ h
        k = bi * P @ z / (1 + bi * z.T.conj() @ P @ z)
        e = R2 - h.T.conj() @ z
        h = h + k @ e.conj()
        P = bi * P - bi * k @ z.T.conj() @ P

        h = h[:,0]
        return e, h, P

    # we could also push vmap to outter function
    fv = jit(vmap(f, in_axes=(None, -1, -1), out_axes=-1))

    h_old = h

    e, h, P = fv(u, h, P)

    outputs = (e, h_old)

    params = (h, P, R2, beta)

    return params, outputs


def cma_2sec(y_f, h_init, w_init, mu1=1e-4, mu2=1e-1, const=comm.const("16QAM", norm=True), device=cpus[0]):

    R2 = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    params = (h_init, w_init, mu1, mu2, R2)
    inputs = (y_f,)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_cma_2sec, params, inputs)
    return ret


def mimo_2sec(h, w, u):
    c = jnp.array([[w[0], w[1]], [-w[1].conj(), w[0].conj()]])
    z = jnp.einsum('ij,ij->j', h, u)
    v = jnp.einsum('ij,j->i', c, z)
    return v


@jit
def step_cma_2sec(params, inputs):
    h, w, mu1, mu2, R2 = params
    u, = inputs

    def loss_fn(P, u):
        h, w = P
        v = mimo_2sec(h, w, u)
        l = jnp.sum(jnp.abs(R2 - jnp.abs(v)**2))
        return l

    P = (h, w)
    l, G = value_and_grad(loss_fn)(P, u)

    outputs = (l, h, w)

    g1, g2 = G
    h = h - mu1 * g1.conj()
    w = w - mu2 * g2.conj()

    params = (h, w, mu1, mu2, R2)

    return params, outputs


def rde(signal, w_init, data=None, train=None, lr=1e-4, const=comm.const("16QAM", norm=True), device=cpus[0]):
    '''
    References:
    [1] Fatadin, I., Ives, D. and Savory, S.J., 2009. Blind equalization and carrier phase recovery
        in a 16-QAM optical coherent system. Journal of lightwave technology, 27(15), pp.3042-3049.
    '''
    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0], signal.shape[-1]), 0, dtype=signal.dtype)
    else:
        if train.shape[0] != signal.shape[0] or data.shape[0] != signal.shape[0]:
            raise ValueError('invalid shape')

    Rs = np.unique(np.abs(const))

    params = (w_init, jnp.array([0j, 0j]), lr, Rs)
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_rde, params, inputs)
    return ret


@jit
def step_rde(params, inputs):
    w, b, lr, Rs = params
    u, x, train = inputs

    l, (gw, gb) = value_and_grad(loss_rde)((w, b), u, Rs, x, train)

    outputs = (l, w, b)

    w = w - lr * gw.conj()
    b = b - 0. * gb.conj()

    params = (w, b, lr, Rs)

    return params, outputs


def loss_rde(p, u, Rs, x, train):
    w, b = p
    v = (mimo(w, u) + b)[None,:]

    R2 = lax.cond(
        train,
        None,
        lambda _: jnp.abs(x)**2,
        None,
        lambda _: Rs[jnp.argmin(jnp.abs(Rs[:,None] * v / jnp.abs(v) - v), axis=0)]**2
    )

    l = jnp.sum(jnp.abs(R2 - jnp.abs(v[0,:])**2))
    return l


def mu_cma(y_f, w_init, lr=1e-4, alpha=0.9999, const=comm.const("16QAM", norm=True), device=cpus[0]):
    d = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    ntap = w_init.shape[-1]
    nch = y_f.shape[-1]
    z  = jnp.zeros((ntap, nch), dtype=y_f.dtype)
    c  = jnp.zeros((nch, nch, ntap), dtype=y_f.dtype)

    y_f = device_put(y_f, device)
    w_init = device_put(w_init, device)
    lr = device_put(lr, device)
    alpha = device_put(alpha, device)
    d = device_put(d, device)

    _, w = scan(step_mu_cma, (w_init, d, c, z, alpha, lr), y_f)
    return w


def loss_mu_cma(w, d, c, u):

    c_sqsum = jnp.sum(jnp.abs(c)**2, axis=-1)
    xcorr_l = jnp.sum(c_sqsum) - jnp.sum(jnp.diag(c_sqsum))

    cma_l = loss_cma(w, u, d)

    return  cma_l + 2 * xcorr_l


@jit
def step_mu_cma(carry, u):
    (w, d, c, z, a, lr) = carry
    v = mimo(w, u)[None,:]
    z = jnp.concatenate((v, z[:-1,:]))

    nch = z.shape[-1]
    z0 = jnp.repeat(z, nch, axis=-1)
    z1 = jnp.tile(z, (1, nch))
    corr_fn = vmap(lambda z0i, z1i: xop.correlate_fft(z0i, z1i), in_axes=-1, out_axes=0)
    corr_flat = corr_fn(z0, z1)
    corr = jnp.reshape(corr_flat, c.shape)
    c = a * c + (1 - a) * corr

    g = grad(loss_mu_cma)(w, d, c, u).conj()
    w = w - lr * g
    return (w, d, c, z, a, lr), w


def lms_cpane(signal, w_init, data=None, train=None, lr=1e-4, beta=0.7, const=comm.const("16QAM", norm=True), device=cpus[0]):
    const = comm.const("16QAM", norm=True)

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0],), 0, dtype=const.dtype)

    dims = signal.shape[-1]

    params_lms = (w_init, lr)
    params_cpane = tuple(map(lambda x: np.tile(x, dims), [1e-5 * (1.+1j), 1e-2 * (1.+1j), 0j, 1j, 0j, beta])) + (const,)
    params = (params_lms, params_cpane)
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_lms_cpane, params, inputs)

    return ret


@jit
def step_lms_cpane(params, inputs):

    u, x, train = inputs

    params_lms, params_cpane = params
    w, lr = params_lms

    #const = params_cpane[-1]

    v = mimo(w, u)

    paxes = (-1,) * 6 + (None,)
    step_cpane_ekf_v = vmap(lambda par, inp: step_cpane_ekf(par, inp), in_axes=(paxes, (-1, -1, None)), out_axes=(paxes, -1))

    params_cpane, outputs_cpane= step_cpane_ekf_v(params_cpane, (v, x, train))

    psi_hat, d = outputs_cpane

    t = v * jnp.exp(-1j * psi_hat.real)

    d = lax.cond(
        train,
        None,
        lambda _: x, # data-aided mode
        None,
        lambda _: d, # const[jnp.argmin(jnp.abs(const[:,None] - t[None,:]), axis=0)]
    )

    r = t - d

    l = jnp.abs(r)**2

    outputs = (l, w, psi_hat)

    g_w = r[..., None, None] * u.conj().T[None,...]

    w = w - lr * g_w

    params = ((w, lr), params_cpane)

    return params, outputs


def cpane_ekf(signal, data=None, train=None, beta=0.8, device=cpus[0]):
    '''
    References:
    [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
    of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
    '''
    const = comm.const("16QAM", norm=True)

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0],), 0, dtype=const.dtype)

    params = (1e-5 * (1.+1j), 1e-2 * (1.+1j), 0j, 1j, 0j, beta, const)
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    psi_hat = _cpane_ekf(params, inputs)

    return psi_hat


@jit
def _cpane_ekf(params, inputs):
    _, ret = lax.scan(step_cpane_ekf, params, inputs)
    psi_hat = ret[0]

    return psi_hat


@jit
def step_cpane_ekf(params, inputs):
    Q, R, Psi_c, P_c, Psi_a, beta, const = params
    r, x, train = inputs

    Psi_p = Psi_c
    P_p = P_c + Q

    Psi_a = beta * Psi_a + (1 - beta) * Psi_c

    d = lax.cond(
        train,
        None,
        lambda _: x, # data-aided mode
        None,
        lambda _: const[jnp.argmin(jnp.abs(const - r * jnp.exp(-1j * Psi_a)))]) # decision directed mode

    H = 1j * d * jnp.exp(1j * Psi_p)
    K = P_p * H.conj() / (H * P_p * H.conj() + R)
    v = r - d * jnp.exp(1j * Psi_p)

    outputs = (Psi_c, d) # return averaged decision results

    Psi_c = Psi_p + K * v
    P_c = (1. - K * H) * P_p

    params = (Q, R, Psi_c, P_c, Psi_a, beta, const)

    return params, outputs


def cpr_foe_ekf(signal, init_states=None, device=cpus[0]):
    '''
    References:
    [1] Jain, A., Krishnamurthy, P.K., Landais, P. and Anandarajah, P.M., 2017.
        EKF for joint mitigation of phase noise, frequency offset and nonlinearity
        in 400 Gb/s PM-16-QAM and 200 Gb/s PM-QPSK systems. IEEE Photonics Journal,
        9(1), pp.1-10.
    [2] Lin, W.T. and Chang, D.C., 2006, May. The extended Kalman filtering algorithm
        for carrier synchronization and the implementation. In 2006 IEEE International
        Symposium on Circuits and Systems (pp. 4-pp). IEEE.
    [3] Akhlaghi, Shahrokh, Ning Zhou, and Zhenyu Huang. "Adaptive adjustment of noise
        covariance in Kalman filter for dynamic state estimation." 2017 IEEE power & energy
        society general meeting. IEEE, 2017.
    '''

    if init_states is None:
        init_states = (
          jnp.array([[1e-2,  0],
                     [0,  1e-5]]),
          1e-1 * jnp.eye(2),
          jnp.array([[0.],
                     [0.]]),
          1. * jnp.eye(2)
        )

    A = jnp.array([[1, 1],
                   [0, 1]])

    const = comm.const("16QAM", norm=True)

    signal = device_put(signal, device)
    init_states = device_put(init_states, device)
    A = device_put(A, device)
    const = device_put(const, device)

    @jit
    def step(states, r):
        Q, R, x_c, P_c = states

        x_p = A @ x_c
        P_p = A @ P_c @ A.T + Q

        p_p = jnp.exp(1j * x_p[0,0])
        s_hat_p = const[jnp.argmin(jnp.abs(const - r * p_p.conj()))]
        r_hat_p = s_hat_p * p_p

        d = r - r_hat_p
        H = jnp.array([[-r_hat_p.imag, 0],
                       [ r_hat_p.real, 0]])
        I = jnp.array([[d.real],
                       [d.imag]])
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ jnp.linalg.inv(S)
        x_c = x_p + K @ I
        P_c = P_p - K @ H @ P_p

        # adapt Q and R
        beta = .99
        # p_c  = jnp.exp(1j * x_c[0,0])
        # e = r - s_hat_p * p_c
        # e_R = jnp.array([[e.real],
        #                  [e.imag]])
        # R = beta * R + (1 - beta) * (e_R @ e_R.T + H @ P_p @ H.T)
        Q = beta * Q + (1. - beta) * K @ I @ I.T @ K.T

        return (Q, R, x_c, P_c), x_p[:,0]

    _, ret = scan(step, init_states, signal)

    return ret.T


def cpr_ekf(signal, init_states=None, device=cpus[0]):
    Q = 3.0e-5 * np.eye(1)
    R = 2.0e-2 * np.eye(2)
    const = comm.const("16QAM", norm=True)

    P_corr = np.array([[1.]])
    phi_corr = np.array([[0.]])

    init_states = (Q, R, phi_corr, P_corr)

    init_states = device_put(init_states, device)
    const = device_put(const, device)

    @jit
    def step(states, r):
        Q, R, phi_corr, P_corr = states
        phi_pred = phi_corr
        P_pred = P_corr + Q

        phi_pred_C = jnp.exp(1j * phi_pred[0,0])
        s_hat = const[jnp.argmin(jnp.abs(const - r * phi_pred_C.conj()))]
        r_hat_pred = s_hat * phi_pred_C

        H_C = 1j * r_hat_pred
        H = jnp.array([[H_C.real], [H_C.imag]])
        I = jnp.array([[(r - r_hat_pred).real], [(r - r_hat_pred).imag]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ jnp.linalg.inv(S)
        phi_corr = phi_pred + K @ I
        P_corr = P_pred - K @ H @ P_pred

        return (Q, R, phi_corr, P_corr), phi_pred[0,0]

    _, ret = scan(step, init_states, signal)

    return ret


def anf(signal, f0, sr, A=1, phi=0, lr=1e-4, device=cpus[0]):
    if jnp.iscomplexobj(signal):
        signal = jnp.stack([signal.real, signal.imag], axis=-1)
        signal = vmap(_anf, in_axes=(-1,) + (None,) * 6, out_axes=(-1,))(
            signal, f0, sr, A, phi, lr, device
        )
        signal = signal[...,0] + jnp.array(1j) * signal[...,1]
    else:
        signal = _anf(signal, f0, sr, A, phi, lr, device)

    return signal


def _anf(signal, f0, sr, A, phi, lr, device):
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

    K = np.arange(signal.shape[0])

    ref = np.array([A * np.cos(w0 * K * T + phi), A * np.sin(w0 * K * T + phi)]).T

    w_init = jnp.array([0., 0.])

    params = (w_init, lr)
    inputs = (signal, ref)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_anf, params, inputs)

    return ret[0]


@jit
def step_anf(params, inputs):
    w, mu = params
    d, x = inputs

    y = jnp.inner(w, x)

    e = d - y

    outputs = (e,)

    w += 2 * mu * e * x

    params = (w, mu)

    return params, outputs


def ddlms(mu_w=1/2**6, mu_f=1/2**7, mu_s=0., mu_b=1/2**12, grad_max=(50., 50.), eps=1e-8,
          vss=True, rho=1e-4, const=comm.const("16QAM", norm=True), lockgain=False):
    '''
    Enhancements
    [1] add bias term to handle varying DC component
    References:
    [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of
        finite-impulse-response filters tolerant to carrier-phase fluctuations
        in digital coherent optical receivers for higher-order quadrature
        amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    [2] Mathews, V. John, and Zhenhua Xie. "A stochastic gradient adaptive filter with
        gradient adaptive step size." IEEE transactions on Signal Processing 41.6 (1993): 2075-2087.
    '''
    const = jnp.asarray(const)

    def init(taps=19, dims=2, dtype=jnp.complex64, mimoinit='zeros'):
        w0 = mimoinitializer(taps, dims, dtype, mimoinit)
        f0 = jnp.full((dims,), 1., dtype=dtype)
        s0 = jnp.full((dims,), 1., dtype=dtype)
        b0 = jnp.full((dims,), 0., dtype=dtype)
        mu_f0 = jnp.full(dims, mu_f)
        mu_s0 = jnp.full(dims, mu_s)
        mu_b0 = jnp.full(dims, mu_b)
        lv0 = jnp.zeros(dims, dtype)
        lk0 = jnp.zeros(dims, dtype)
        lc0 = jnp.zeros(dims, dtype)
        le_f0 = jnp.zeros(dims, dtype)
        le_s0 = jnp.zeros(dims, dtype)
        le_b0 = jnp.zeros(dims, dtype)
        return (w0, f0, s0, b0, mu_f0, mu_s0, mu_b0, lv0, lk0, lc0, le_f0, le_s0, le_b0)

    def update(state, inp):
        w, f, s, b, mu_f, mu_s, mu_b, lv, lk, lc, le_f, le_s, le_b = state
        u, x, train = inp

        v = mimo(w, u)
        k = v * f
        c = k * s
        z = c + b
        d = jnp.where(train, x, const[jnp.argmin(jnp.abs(const[:,None] - z[None,:]), axis=0)])
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

        # clip the grads of f and s which are less regulated than w,
        # it may stablize this algo. in some corner cases?
        gf = jnp.where(jnp.abs(gf) > grad_max[0], gf / jnp.abs(gf) * grad_max[0], gf)
        gs = jnp.where(jnp.abs(gs) > grad_max[1], gs / jnp.abs(gs) * grad_max[1], gs)

        out = ((w, f, s, b), (l, d, mu_f, mu_s, mu_b))

        # auto stepsize adjustment[2]
        if vss:
            mu_f_ub = 0.8 * 2 / (jnp.abs(v)**2)
            mu_f += mu_f / (mu_f + eps) * rho * 2 * (e_f * le_f.conj() * lv.conj() * v).real
            mu_f = jnp.minimum(mu_f, mu_f_ub)
            mu_s_ub = 0.8 * 2 / (jnp.abs(k)**2)
            mu_s += mu_s / (mu_s + eps) * rho * 2 * (e_s * le_s.conj() * lk.conj() * k).real
            mu_s = jnp.minimum(mu_s, mu_s_ub)
            # mu_b_ub = 0.8 * 2 / (jnp.abs(c)**2)
            # mu_b += mu_b / (mu_b + eps) * 1e-4 * 2 * (e_b * le_b.conj() * lc.conj() * c).real
            # mu_b = jnp.minimum(mu_b, mu_b_ub)

        # update
        w = w - mu_w * gw
        f = f - mu_f * gf
        s = s - mu_s * gs
        b = b - mu_b * gb

        if lockgain:
            w *= (jnp.abs(f) * jnp.abs(s))[:, None, None]
            w /= (jnp.sqrt(jnp.sum(jnp.abs(w)**2, axis=(1, 2))))[:, None, None]
            f /= jnp.abs(f)
            s /= jnp.abs(s)

        state = (w, f, s, b, mu_f, mu_s, mu_b, v, k, c, e_f, e_s, e_b)

        return state, out

    def static_map(ps, yf):
        ws, fs, ss, bs = ps
        return jax.vmap(mimo)(ws, yf) * fs * ss + bs

    return AdaptiveFilter(init, update, static_map)

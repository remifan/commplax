import numpy as np
from jax import grad, value_and_grad, lax, jit, vmap, numpy as jnp, devices, device_put
from jax.ops import index, index_add, index_update
from functools import partial
from commplax import xop, src


cpus = devices("cpu")
gpus = devices("gpu")


def dbp_model_general(y, steps, h, c):

    y = device_put(y)
    h = device_put(h)
    c = device_put(c)

    D = jit(vmap(lambda y,h: xop.conv1d_fft_oa(y, h, mode='SAME'), in_axes=1, out_axes=1))
    # D = jit(vmap(lambda y,h: xop.conv1d_lax(y, h), in_axes=1, out_axes=1)) # too slow
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(steps):
        y = D(y, h[i])
        y = N(y, c[i])

    return y


def dbp_model_direct(y, steps, H, c):
    y = device_put(y)
    H = device_put(H)
    c = device_put(c)

    fft = lambda x: jnp.fft.fft(x, axis=0)
    ifft = lambda x: jnp.fft.ifft(x, axis=0)

    D = jit(lambda y,H: ifft(fft(y) * H))
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(steps):
        y = D(y, H[i])
        y = N(y, c[i])

    return y


def foe_4s(x, sr=2*jnp.pi):
    X4 = jnp.fft.fft(x**4, axis=0)
    h  = jnp.abs(X4)**2
    f  = jnp.argmax(h, axis=0)
    N  = len(h)

    f = jnp.where(f >= N / 2, f - N, f)

    fo_hat = f / N * sr / 4

    return fo_hat, h


def scan(f, init, xs, unroll=1):
    '''
    "NOTE: ``scan`` is known to cause memory leaks when not called within a jitted"
    "https://github.com/google/jax/issues/3158#issuecomment-631851006"
    "https://github.com/google/jax/pull/5029/commits/977c9c40efa378d1321a7dd8c712af528939ed5f"
    "https://github.com/google/jax/pull/5029"
    "NOTE": ``scan`` runs much slower on GPU than CPU if loop iterations are small
    "https://github.com/google/jax/issues/2491"
    "https://github.com/google/jax/pull/3076"
    '''
    @partial(jit, static_argnums=(0,3))
    def _scan(f, init, xs, unroll):
        return lax.scan(f, init, xs, unroll=unroll)

    return _scan(f, init, xs, unroll)


def mimo(w, u):
    v = jnp.einsum('ijt,tj->i', w, u) # no dimension axis
    return v


def mimo_bias(w, b, u):
    v = jnp.einsum('ijt,tj->i', w, u) + b # no dimension axis
    return v


def dd_lms(signal, w_init, data=None, train=None, lr_w=1/2**10, lr_f=1/2**6, lr_s=0.,
           grad_max=(50., 50.), const=src.const("16QAM", norm=True), device=cpus[0]):
    '''
    Impl. follows Fig. 6 in [1]
    References:
    [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of finite-impulse-response
    filters tolerant to carrier-phase fluctuations in digital coherent optical receivers for
    higher-order quadrature amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    '''

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0], signal.shape[-1]), 0, dtype=const.dtype)
    else:
        if train.shape[0] != signal.shape[0] or data.shape[0] != signal.shape[0]:
           raise ValueError('invalid shape')

    dims = signal.shape[-1]
    f_init = np.full((dims,), 1+0j, dtype=const.dtype) # dummy initial value
    s_init = np.full((dims,), 1+0j, dtype=const.dtype) # dummy initial value
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


def cma(y_f, w_init, bias=False, lr=1e-4, R2=1.32, device=cpus[0]):

    #const = src.const("16QAM", norm=True)
    #R2 = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)
    b_init = np.zeros(y_f.shape[-1], dtype=y_f.dtype)

    params = (w_init, b_init, bias, R2, lr)
    inputs = (y_f,)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_cma, params, inputs)

    if not bias:
        ret = ret[:2]

    return ret


@jit
def step_cma(params, inputs):
    w, b, bias, R2, lr = params
    u, = inputs

    l, (gw, gb) = value_and_grad(loss_cma)((w, b), u, R2)

    outputs = (l, w, b)

    w = w - lr * gw.conj()

    b = lax.cond(
        bias,
        None,
        lambda _: b - 1e-4 * gb.conj(),
        None,
        lambda _: b
    )

    params = (w, b, bias, R2, lr)

    return params, outputs


def loss_cma(P, u, R2):
    w, b = P
    v = mimo(w, u) + b
    l = jnp.sum(jnp.abs(R2 - jnp.abs(v)**2))
    return l


def rls_cma(y_f, w_init, beta=0.9999, delta=1e-4, const=src.const("16QAM", norm=True), device=cpus[0]):
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

    _, ret = scan(step_rls_cma, params, inputs, unroll=8)

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
    bi = 1 / beta

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


def cma_2sec(y_f, h_init, w_init, mu1=1e-4, mu2=1e-1, const=src.const("16QAM", norm=True), device=cpus[0]):

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


def rde(signal, w_init, data=None, train=None, lr=1e-4, const=src.const("16QAM", norm=True), device=cpus[0]):
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

    params = (w_init, lr, Rs)
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, ret = scan(step_rde, params, inputs)
    return ret


@jit
def step_rde(params, inputs):
    w, lr, Rs = params
    u, x, train = inputs

    l, g = value_and_grad(loss_rde)(w, u, Rs, x, train)

    outputs = (l, w)

    w = w - lr * g.conj()

    params = (w, lr, Rs)

    return params, outputs


def loss_rde(w, u, Rs, x, train):
    v = mimo(w, u)[None,:]

    R2 = lax.cond(
        train,
        None,
        lambda _: jnp.abs(x)**2,
        None,
        lambda _: Rs[jnp.argmin(jnp.abs(Rs[:,None] * v / jnp.abs(v) - v), axis=0)]**2
    )

    l = jnp.sum(jnp.abs(R2 - jnp.abs(v[0,:])**2))
    return l


def mu_cma(y_f, w_init, lr=1e-4, alpha=0.9999, const=src.const("16QAM", norm=True), device=cpus[0]):
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


def lms_cpane(signal, w_init, data=None, train=None, lr=1e-4, beta=0.7, const=src.const("16QAM", norm=True), device=cpus[0]):
    const = src.const("16QAM", norm=True)

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
    const = src.const("16QAM", norm=True)

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
    '''

    if init_states is None:
        init_states = (
          jnp.array([[1e-3,  0],
                     [0,  1e-7]]),
          1e-2 * jnp.eye(2),
          jnp.array([[0.],
                    [0.]]),
          1. * jnp.eye(2)
        )

    A = jnp.array([[1, 1],
                   [0, 1]])

    const = src.const("16QAM", norm=True)

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
        s_hat = const[jnp.argmin(jnp.abs(const - r * p_p.conj()))]
        r_hat_p = s_hat * p_p

        H = jnp.array([[-r_hat_p.imag, 0],
                      [ r_hat_p.real, 0]])
        I = jnp.array([[(r - r_hat_p).real],
                      [(r - r_hat_p).imag]])
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ jnp.linalg.inv(S)
        x_c = x_p + K @ I
        P_c = P_p - K @ H @ P_p

        return (Q, R, x_c, P_c), x_p[:,0]

    _, ret = scan(step, init_states, signal)

    return ret.T


def cpr_ekf(signal, init_states=None, device=cpus[0]):
    Q = 3.0e-5 * np.eye(1)
    R = 2.0e-2 * np.eye(2)
    const = src.const("16QAM", norm=True)

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

    return ret


@jit
def step_anf(params, inputs):
    w, mu = params
    d, x = inputs

    y = jnp.inner(w, x)

    e = d - y

    outputs = (e, y)

    w += 2 * mu * e * x

    params = (w, mu)

    return params, outputs


def measure_cd(x, sr, start=-0.25, end=0.25, bins=1000, wavlen=1550e-9):
    '''
    References:
        Zhou, H., Li, B., Tang, et. al, 2016. Fractional fourier transformation-based
        blind chromatic dispersion estimation for coherent optical communications.
        Journal of Lightwave Technology, 34(10), pp.2371-2380.
    '''
    c = 299792458.
    p = jnp.linspace(start, end, bins)
    N = x.shape[0]
    K = p.shape[0]

    L = jnp.zeros(K, dtype=jnp.float32)

    def f(_, pi):
        return None, jnp.sum(jnp.abs(xop.frft(jnp.abs(xop.frft(x, pi))**2, -1))**2)

    # Use `scan` instead of `vmap` here to avoid potential large memory allocation.
    # Despite the speed of `scan` scales surprisingly well to large bins,
    # the speed has a lowerbound e.g 600ms at bins=1, possiblely related to the blind
    # migration of `frft` from Github :) (could frft be jitted in theory?).
    # TODO review `frft`
    _, L = scan(f, None, p)

    B2z = jnp.tan(jnp.pi/2 - (p - 1) / 2 * jnp.pi)/(sr * 2 * jnp.pi / N * sr)
    Dz_set  = -B2z / wavlen**2 * 2 * jnp.pi * c # the swept set of CD metrics
    Dz_hat = Dz_set[jnp.argmin(L)] # estimated accumulated CD

    return Dz_hat, L, Dz_set


def getpower(x):
    return jnp.mean(jnp.abs(x)**2, axis=0)


def scale_rotate(y, x, testing_phases=4, device=gpus[0]):

    y = device_put(y, device)
    x = device_put(x, device)

    y = y / jnp.sqrt(getpower(y) / getpower(x))

    TP = jnp.exp(1j * 2 * jnp.pi / testing_phases * jnp.arange(testing_phases))

    def f(y, x):
        y_t = jnp.outer(y, TP)
        x_t = jnp.tile(x[:,None], (1, testing_phases))
        snr_t = 10. * jnp.log10(getpower(y_t) / getpower(y_t - x_t))
        return TP[jnp.argmax(snr_t)]

    return y * jit(vmap(f, in_axes=-1))(y, x)


def corr_local(y, x, frame_size=2000, L=None, device=gpus[0]):
    y = device_put(y, device)
    x = device_put(x, device)

    if L is None:
        L = len(np.unique(x))

    Y = xop.frame(y, frame_size, frame_size, True)
    X = xop.frame(x, frame_size, frame_size, True)

    lag = jnp.arange(-(frame_size-1)//2, (frame_size+1)//2)

    corr_v = vmap(lambda a, b: xop.correlate_fft(a, b), in_axes=-1, out_axes=-1)

    def f(_, z):
        y, x = z
        c = jnp.abs(corr_v(y, x))
        return _, lag[jnp.argmax(c, axis=0)]

    _, ret = scan(f, None, (Y, X))

    return ret


def foe_local(y, frame_size=65536, frame_step=100, sps=1, device=cpus[0]):
    '''
    resolution = samplerate / N / 4 / sps (linear interp.)
    '''
    y = device_put(y, device)

    return _foe_local(y, frame_size, frame_step, sps)


@partial(jit, static_argnums=(1, 2, 3))
def _foe_local(y, frame_size, frame_step, sps):

    Y = xop.frame(y, frame_size, frame_step, True)

    N = y.shape[0]
    frames = Y.shape[0]

    def foe(carray, y):
        fo_hat, _ = foe_4s(y)
        return carray, fo_hat

    _, fo_hat = scan(foe, None, Y)

    xp = jnp.arange(frames) * frame_step + frame_size//2
    x = jnp.arange(N * sps) / sps
    fo_hat /= sps

    interp = vmap(lambda x, xp, fp: jnp.interp(x, xp, fp), in_axes=(None, None, -1), out_axes=-1)

    fo_hat_ip = interp(x, xp, fo_hat)

    return fo_hat_ip


def power_local(y, frame_size=2000, frame_step=100, sps=1, device=cpus[0]):
    y = device_put(y, device)

    return _power_local(y, frame_size, frame_step, sps)


@partial(jit, static_argnums=(1, 2, 3))
def _power_local(y, frame_size, frame_step, sps):
    yf = xop.frame(y, frame_size, frame_step, True)

    N = y.shape[0]
    frames = yf.shape[0]

    _, power = scan(lambda c, y: (c, jnp.mean(jnp.abs(y)**2, axis=0)), None, yf)

    xp = jnp.arange(frames) * frame_step + frame_size//2
    x = jnp.arange(N * sps) / sps

    interp = vmap(lambda x, xp, fp: jnp.interp(x, xp, fp), in_axes=(None, None, -1), out_axes=-1)

    power_ip = interp(x, xp, power)

    return power_ip


def getpower(x, real=False):
    ''' get signal power '''
    if real:
        return jnp.mean(x.real**2, axis=0), jnp.mean(x.imag**2, axis=0)
    else:
        return jnp.mean(abs(x)**2, axis=0)


def normpower(x, real=False):
    ''' normalize signal power '''
    if real:
        pr, pi = getpower(x, real=True)
        return x.real / jnp.sqrt(pr) + 1j * x.imag / jnp.sqrt(pi)
    else:
        return x / jnp.sqrt(getpower(x))



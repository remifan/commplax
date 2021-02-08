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
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(steps):
        y = D(y, h[i])
        y = N(y, c[i])

    return y


def tddbp_2d(y, h, c):
    niter = len(c)

    y = device_put(y)
    h = device_put(h)
    c = device_put(c)

    D = jit(vmap(lambda y,h: xop.conv1d_lax(y, h), in_axes=1, out_axes=1))
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(niter):
        y = D(y, h[i])
        y = N(y, c[i])

    return y


def dbp_model_direct(y, H, c):
    niter = len(c)

    y = device_put(y)
    H = device_put(H)
    c = device_put(c)

    fft = lambda x: jnp.fft.fft(x, axis=0)
    ifft = lambda x: jnp.fft.ifft(x, axis=0)

    D = jit(lambda y,H: ifft(fft(y) * H))
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(niter):
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


@partial(jit, static_argnums=(0,))
def scan(f, init, xs):
    '''
    "NOTE: ``scan`` is known to cause memory leaks when not called within a jitted"
    "https://github.com/google/jax/issues/3158#issuecomment-631851006"
    "https://github.com/google/jax/pull/5029/commits/977c9c40efa378d1321a7dd8c712af528939ed5f"
    "https://github.com/google/jax/pull/5029"
    '''
    return lax.scan(f, init, xs)


def mimo(w, u):
    v = jnp.einsum('ijt,tj->i', w, u) # no dimension axis
    return v


def dd_lms(signal, w_init, data=None, train=None, const=src.const("16QAM", norm=True), device=cpus[0]):
    '''
    Impl. follows Fig. 6 in [1]
    References:
    [1] Mori, Y., Zhang, C. and Kikuchi, K., 2012. Novel configuration of finite-impulse-response
    filters tolerant to carrier-phase fluctuations in digital coherent optical receivers for
    higher-order quadrature amplitude modulation signals. Optics express, 20(24), pp.26236-26251.
    '''

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0],), 0, dtype=const.dtype)
    else:
        if train.shape[0] != signal.shape[0] or data.shape[0] != signal.shape[0]:
            raise ValueError('invalid shape')

    dims = signal.shape[-1]
    f_init = np.full((dims,), const[0], dtype=const.dtype) # dummy initial value
    s_init = np.full((dims,), const[0], dtype=const.dtype) # dummy initial value
    params = (1/2**8, 1/2**7, 1/2**8, f_init, s_init, 1e-12, w_init, const)
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
    mu_p, mu_f, mu_s, f, s, eps, w, const = params
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

    dd_err = x - d

    psi_hat = jnp.abs(f)/f * jnp.abs(s)/s
    e_p = d * psi_hat  - v
    e_f = d - f * v
    e_s = d - s * f  * v

    outputs = (w, psi_hat, dd_err)

    # update
    s = s + mu_s / (jnp.abs(f * v)**2 + eps) * e_s * (f * v).conj()
    f = f + mu_f / (jnp.abs(v)**2 + eps) * e_f * v.conj()
    grad = e_p[:, None, None] * u.conj().T[None, ...]
    w = w + mu_p * grad

    params = (mu_p, mu_f, mu_s, f, s, eps, w, const)

    return params, outputs


def cma(y_f, w_init, lr=1e-4, const=src.const("16QAM", norm=True), device=cpus[0]):

    d = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    y_f = device_put(y_f, device)
    w_init = device_put(w_init, device)
    lr = device_put(lr, device)
    d = device_put(d, device)

    _, ret = scan(step_cma, (w_init, d, lr), y_f)
    return ret


@jit
def step_cma(carry, u):
    w, d, lr = carry
    l, g = value_and_grad(loss_cma)(w, u, d)
    w = w - lr * g.conj()
    return (w, d, lr), (l, w)


def loss_cma(w, u, d):
    v = mimo(w, u)
    l = jnp.sum(jnp.abs(d - jnp.abs(v)**2))
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


def lms_cpane(signal, w_init, data=None, train=None, lr=1e-3, beta=0.9, device=cpus[0]):
    const = src.const("16QAM", norm=True)

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0],), 0, dtype=const.dtype)

    params_lms = (w_init, lr)
    params_cpane = (1e-4 * (1.+1j), 1e-2 * (1.+1j), 0j, 1j, 0j, beta, const)
    params = params_lms + params_cpane
    inputs = (signal, data, train)

    params = device_put(params, device)
    inputs = device_put(inputs, device)

    _, w = step_lms_cpane(params, inputs)

    return w


@jit
def step_lms_cpane(params, inputs):

    u, x, train = inputs

    params_lms, params_cpane = (params[:2], params[2:])
    w, lr = params_lms

    v = mimo(w, u)[None,:]

    params_cpane, outputs_cpane= step_cpane_ekf(params_cpane, (v, x, train))

    psi_hat = outputs_cpane[0]

    t = v * jnp.exp(-1j * psi_hat)

    l, g = value_and_grad(loss_lms)(t, x)
    w = w - lr * g.conj()

    params = (w, lr) + params_cpane

    return params, (l, w)


def loss_lms(v, x):
    return jnp.sum(jnp.abs(x - v)**2)


def cpane_ekf(signal, data=None, train=None, beta=0.9, device=cpus[0]):
    '''
    References:
    [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
    of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
    '''
    const = src.const("16QAM", norm=True)

    if train is None:
        train = np.full((signal.shape[0],), False)
        data = np.full((signal.shape[0],), 0, dtype=const.dtype)

    params = (1e-4 * (1.+1j), 1e-2 * (1.+1j), 0j, 1j, 0j, beta, const)
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

    outputs = (Psi_c,)

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
          jnp.array([[1e-4,  0],
                     [0,  1e-8]]),
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



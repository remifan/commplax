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


def mimo(w, u):
    v = jnp.einsum('ijt,tj->i', w, u)
    return v


def cma(y_f, w_init, lr=1e-4, const=src.const("16QAM", norm=True), device=cpus[0]):

    d = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    y_f = device_put(y_f, device)
    w_init = device_put(w_init, device)
    lr = device_put(lr, device)
    d = device_put(d, device)

    _, w = lax.scan(step_cma, (w_init, d, lr), y_f)
    return w


@jit
def step_cma(carry, u):
    w, d, lr = carry
    g = grad(loss_cma)(w, u, d).conj()
    w = w - lr * g
    return (w, d, lr), w


def loss_cma(w, u, d):
    v = mimo(w, u)
    l = jnp.sum(jnp.abs(d - jnp.abs(v)**2))
    return l


def mu_cma(y_f, w_init, lr=1e-4, alpha=0.999, const=src.const("16QAM", norm=True), device=cpus[0]):
    d = jnp.mean(abs(const)**4) / jnp.mean(abs(const)**2)

    ntap = w_init.shape[0]
    nch = y_f.shape[-1]
    z  = jnp.zeros((ntap, nch), dtype=y_f.dtype)
    c  = jnp.zeros((nch, nch, ntap, ntap), dtype=y_f.dtype)

    y_f = device_put(y_f, device)
    w_init = device_put(w_init, device)
    lr = device_put(lr, device)
    alpha = device_put(alpha, device)
    d = device_put(d, device)

    _, w = lax.scan(step_mu_cma, (w_init, d, c, z, alpha, lr), y_f)
    return w


@jit
def step_mu_cma(carry, u):
    (w, d, c, z, a, lr) = carry
    v = mimo(w, u)[None,:]
    z = jnp.concatenate((v, z[:-1,:]))
    g = grad(loss_mu_cma)(w, d, c, z, a, u).conj()
    w = w - lr * g
    return (w, d, c, z, a, lr), w


def loss_mu_cma(w, d, c, z, a, u):
    nch = z.shape[-1]
    z0 = jnp.repeat(z, nch, axis=-1)
    z1 = jnp.tile(z, (1, nch))

    corr_fn = vmap(lambda z0i, z1i: jnp.outer(z0i, z1i.conj()), in_axes=-1, out_axes=0)

    corr_flat = corr_fn(z0, z1)

    corr = jnp.reshape(corr_flat, (nch,) * 4)

    c = a * c + (1 - a) * corr
    c_l2norm = jnp.abs(c)**2
    corr_suml2norm = jnp.sum(c_l2norm, axis=(-2,-1))
    sum_xcorr = jnp.sum(corr_suml2norm) - jnp.sum(jnp.diag(corr_suml2norm))

    cma_l = loss_cma(w, u, d)

    return  cma_l + sum_xcorr


def loss_lms(v, x, np=jnp):
    return np.sum(np.abs(x - v)**2)


def cpane_ekf(signal, beta=0.95, device=cpus[0]):
    '''
    References:
    [1] Pakala, L. and Schmauss, B., 2016. Extended Kalman filtering for joint mitigation
    of phase and amplitude noise in coherent QAM systems. Optics express, 24(6), pp.6391-6401.
    '''
    const = src.const("16QAM", norm=True)

    init_states = (1e-4 * (1.+1j), 1e-2 * (1.+1j), 0j, 1j, 0j)

    signal = device_put(signal, device)
    init_states = device_put(init_states, device)
    const = device_put(const, device)

    @jit
    def step(states, r):
        Q, R, Psi_c, P_c, Psi_a = states

        Psi_p = Psi_c
        P_p = P_c + Q

        Psi_a = beta * Psi_a + (1 - beta) * Psi_c

        t = r * jnp.exp(-1j * Psi_a)
        d = const[jnp.argmin(jnp.abs(const - t))]

        H = 1j * d * jnp.exp(1j * Psi_p)
        K = P_p * H.conj() / (H * P_p * H.conj() + R)
        v = r - d * jnp.exp(1j * Psi_p)
        Psi_c = Psi_p + K * v
        P_c = (1. - K * H) * P_p

        return (Q, R, Psi_c, P_c, Psi_a), Psi_c

    _, ret = lax.scan(step, init_states, signal)

    return ret


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
                     [0,  1e-9]]),
          1e-1 * jnp.eye(2),
          jnp.array([[0.],
                    [0.]]),
          jnp.eye(2)
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

    _, ret = lax.scan(step, init_states, signal)

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
    _, L = lax.scan(f, None, p)

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



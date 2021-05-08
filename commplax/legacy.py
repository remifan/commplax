import numpy as np
from jax import grad, value_and_grad, lax, jit, vmap, numpy as jnp, devices, device_put
from jax.ops import index, index_add, index_update
from functools import partial
from commplax import xop, comm


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

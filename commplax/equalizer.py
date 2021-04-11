import numpy as np
from jax import jit, device_put, numpy as jnp, lax
from commplax import xop, comm, xcomm, util, adaptive_filter as af


def tddbp(signal, sr, lp, dist, spans, taps, xi=0.6, D=16.5E-6, polmux=True, backend=util.gpufirstbackend()):
    x = device_put(signal)
    return jit(_tddbp,
               static_argnums=(4, 5, 8,),
               backend=backend)(x, sr, lp, dist, spans, taps, xi, D, polmux)


def _tddbp(x, sr, lp, dist, spans, taps, xi, D, polmux):
    x = xcomm.normpower(x)
    powscale = np.sqrt(2) if polmux else 1.
    _, param_D, param_N = xcomm.dbp_params(sr, dist/spans, spans, taps, fiber_dispersion=D, polmux=polmux)
    return xcomm.dbp_timedomain(x / powscale, param_D,
                                xi * 10.**(lp / 10 - 3) * param_N) * powscale


def cdcomp(signal, sr, CD, fc=193.4e12, polmux=True):
    pi = np.pi
    C = 299792458. # speed of light [m/s]
    D = 16.5E-6
    dist = CD / D
    lambda_ = C / fc
    # estimate minimal required number of taps for single step TDDBP
    mintaps = int(np.ceil(2 * pi * CD * lambda_**2 / (2 * pi * C) * sr**2))
    return tddbp(signal, sr, 0., dist, 1, mintaps, xi=0., D=D, polmux=polmux)


def modulusmimo(signal, sps=2, taps=19, cma_samples=20000, modformat='16QAM', const=None, backend='cpu'):
    '''
    Adaptive MIMO equalizer for M-QAM signal
    '''
    y = jnp.asarray(signal)

    if y.shape[0] < cma_samples:
        raise ValueError('cam_samples must > given samples length')

    if const is None:
        const = comm.const(modformat, norm=True)

    R2 = np.array(np.mean(abs(const)**4) / np.mean(abs(const)**2))
    Rs = np.array(np.unique(np.abs(const)))

    return jit(_modulusmimo,
               static_argnums=(3, 4, 5,),
               backend=backend)(y, R2, Rs, sps, taps, cma_samples)


def _modulusmimo(y, R2, Rs, sps, taps, cma_samples):
    # prepare adaptive filters

    y = jnp.asarray(y)

    dims = y.shape[-1]
    cma_init, cma_update, _ = af.mucma(R2=R2, dims=dims)
    rde_init, rde_update, rde_map = af.rde(Rs=Rs, dims=dims)

    # framing signal to enable parallelization (a.k.a `jax.vmap`)
    yf = af.frame(y, taps, sps)

    # get initial weights
    s0 = cma_init(taps=taps, dtype=y.dtype)
    # initialize MIMO via MU-CMA to avoid singularity
    (w0, *_,), (ws1, loss1) = af.iterate(cma_update, s0, yf[:cma_samples])
    # switch to RDE
    _, (ws2, loss2) = af.iterate(rde_update, w0, yf[cma_samples:])
    loss = jnp.concatenate([loss1, loss2], axis=0)
    ws = jnp.concatenate([ws1, ws2], axis=0)
    x_hat = rde_map(ws, yf)

    return x_hat, loss, ws


def lmsmimo(signal, truth, sps=2, taps=31,
            lrw=1/2**6, lrf=1/2**7, lrs=0., lrb=1/2**12, beta=0.9,
            backend='cpu'):
    return jit(_lmsmimo,
               static_argnums=(2, 3),
               backend=backend)(signal, truth, sps, taps, lrw, lrf, lrs, lrb, beta)


def _lmsmimo(signal, truth, sps, taps, mu_w, mu_f, mu_s, mu_b, beta):
    y = jnp.asarray(signal)
    x = jnp.asarray(truth)
    lms_init, lms_update, lms_map = af.ddlms(mu_w=mu_w,
                                             mu_f=mu_f,
                                             mu_s=mu_s,
                                             mu_b=mu_b,
                                             beta=beta,
                                             lockgain=False)
    yf = af.frame(y, taps, sps)
    s0 = lms_init(taps, mimoinit='centralspike')
    _, (ss, (loss, *_)) = af.iterate(lms_update, s0, yf, x)
    xhat = lms_map(ss, yf)
    return xhat, loss


def rdemimo(signal, truth, lr=1/2**13, sps=2, taps=31, backend='cpu',
            const=comm.const("16QAM", norm=True)):
    Rs = np.array(np.unique(np.abs(const)))
    return jit(_rdemimo,
               static_argnums=(4, 5),
               backend=backend)(signal, truth, lr, Rs, sps, taps)


def _rdemimo(signal, truth, lr, Rs, sps, taps):
    y = jnp.asarray(signal)
    x = jnp.asarray(truth)
    dims = y.shape[-1]
    rde_init, rde_update, rde_map = af.rde(lr=lr, Rs=Rs, dims=dims)
    yf = af.frame(y, taps, sps)
    s0 = rde_init(taps=taps, dtype=y.dtype)
    _, (ss, loss) = af.iterate(rde_update, s0, yf, x)
    xhat = rde_map(ss, yf)
    return xhat, loss


def qamfoe(x, local=False, fitkind=None, lfoe_fs=16384, lfoe_st=5000, backend='cpu'):
    x = device_put(x)
    return jit(_qamfoe,
               static_argnums=(1, 2, 3, 4),
               backend=backend)(x, local, fitkind, lfoe_fs, lfoe_st)


def _qamfoe(x, local, fitkind, lfoe_fs, lfoe_st):
    dims = x.shape[-1]

    if local:
        fo = xcomm.localfoe(x, frame_size=lfoe_fs, frame_step=lfoe_st, sps=1, fitkind=fitkind).mean(axis=-1)
        x *= jnp.tile(jnp.exp(-1j * jnp.cumsum(fo))[:, None], (1, dims))
    else:
        fo = xcomm.foe_mpowfftmax(x)[0].mean()
        T = jnp.arange(x.shape[0])
        x *= jnp.tile(jnp.exp(-1j * fo * T)[:, None], (1, dims))

    return x, fo


def ekfcpr(signal, truth=None, modformat='16QAM', const=None, backend='cpu'):
    y = jnp.asarray(signal)
    x = jnp.asarray(truth)
    if const is None:
        const=comm.const(modformat, norm=True)
    const = jnp.asarray(const)
    return jit(_ekfcpr, backend=backend)(y, x, const)


def _ekfcpr(y, x, const):
    dims = y.shape[-1]
    cpr_init, cpr_update, cpr_map = af.array(af.cpane_ekf, dims)(beta=0.6, const=const)
    cpr_state = cpr_init()
    _, (phi, _) = af.iterate(cpr_update, cpr_state, y, x)
    xhat = cpr_map(phi, y)
    return xhat, phi



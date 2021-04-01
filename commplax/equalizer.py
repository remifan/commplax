import numpy as np
from jax import jit, device_put, numpy as jnp
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


def qammimo(signal, sps=2, taps=19, cma_samples=20000, modformat='16QAM', const=None, backend='cpu'):
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

    return jit(_qammimo,
               static_argnums=(3, 4, 5,),
               backend=backend)(y, R2, Rs, sps, taps, cma_samples)


def _qammimo(y, R2, Rs, sps, taps, cma_samples):
    # prepare adaptive filters

    y = jnp.asarray(y)

    dims = y.shape[-1]
    cma_init, cma_update, _ = af.mucma(R2=R2, dims=dims)
    rde_init, rde_update, rde_map = af.rde(Rs=Rs, dims=dims)

    rtap = (taps + 1) // sps - 1
    mimo_delay = int(np.ceil((rtap + 1) / sps) - 1)

    # zero-pad to remove filter delay
    y_pad = jnp.pad(y, [[mimo_delay * sps, taps - sps * (mimo_delay + 1)], [0,0]])
    # framing signal to enable parallelization (a.k.a `jax.vmap`)
    yf = jnp.array(xop.frame(y_pad, taps, sps))

    # get initial weights
    s0 = cma_init(taps=taps, dtype=y.dtype)
    # initialize MIMO via MU-CMA to avoid singularity
    (w0, *_,), (loss1, ws1) = af.iterate(cma_update, s0, yf[:cma_samples])
    # switch to RDE
    _, (loss2, ws2) = af.iterate(rde_update, w0, yf[cma_samples:])
    loss = jnp.concatenate([loss1, loss2], axis=0)
    ws = jnp.concatenate([ws1, ws2], axis=0)
    x_hat = rde_map(ws, yf)

    return x_hat, loss, ws


def qamfoe(x, uselocalfoe=False, lfoe_fs=16384, lfoe_st=5000, backend='cpu'):
    x = device_put(x)
    return jit(_qamfoe,
               static_argnums=(1, 2, 3),
               backend=backend)(x, uselocalfoe, lfoe_fs, lfoe_st)


def _qamfoe(x, uselocalfoe, lfoe_fs, lfoe_st):
    dims = x.shape[-1]

    if uselocalfoe:
        fo_local = xcomm.localfoe(x, frame_size=lfoe_fs, frame_step=lfoe_st, sps=1)
        fo_local = jnp.mean(fo_local, axis=-1)
        # xop.polyfit is inaccurate!
        # fo_T = np.arange(len(fo_local))
        # fo_poly = xop.polyfit(fo_T, fo_local, 3)
        # fo_local_poly = jnp.polyval(fo_poly, fo_T)
        # x *= jnp.tile(jnp.exp(-1j * jnp.cumsum(fo_local_poly))[:, None], (1, dims))
        # fo = fo_poly
        x *= jnp.tile(jnp.exp(-1j * jnp.cumsum(fo_local))[:, None], (1, dims))
        fo = fo_local
    else:
        fo, fo_metric = xcomm.foe_mpowfftmax(x)
        fo = jnp.mean(fo)
        T = jnp.arange(x.shape[0])
        x *= jnp.tile(jnp.exp(-1j * fo * T)[:, None], (1, dims))

    return x, fo


def qamcpr(signal, modformat='16QAM', const=None, backend='cpu'):
    x = device_put(signal)
    if const is None:
        const=comm.const(modformat, norm=True)
    const = jnp.asarray(const)
    return jit(_qamcpr, backend=backend)(x, const)


def _qamcpr(x, const):
    dims = x.shape[-1]
    cpr_init, cpr_update, cpr_map = af.array(af.cpane_ekf, dims)(beta=0.7, const=const)
    cpr_state = cpr_init()
    _, (phi, _) = af.iterate(cpr_update, cpr_state, x)
    x = cpr_map(phi, x)
    return x, phi


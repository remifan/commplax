import numpy as np
from jax import grad, value_and_grad, lax, jit, vmap, numpy as jnp, devices, device_put
from jax.ops import index, index_add, index_update
from functools import partial
from commplax import xop, comm

cpus = devices("cpu")
gpus = devices("gpu")


def dbp_timedomain(y, h, c):

    y = device_put(y)
    h = device_put(h)
    c = device_put(c)

    steps = c.shape[0]

    D = jit(vmap(lambda y,h: xop.conv1d_fft_oa(y, h, mode='SAME'), in_axes=1, out_axes=1))
    # D = jit(vmap(lambda y,h: xop.conv1d_lax(y, h), in_axes=1, out_axes=1)) # often too slow for long h
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(steps):
        y = D(y, h[i])
        y = N(y, c[i])

    return y


def dbp_direct(y, H, c):
    y = device_put(y)
    H = device_put(H)
    c = device_put(c)

    steps = c.shape[0]

    fft = lambda x: jnp.fft.fft(x, axis=0)
    ifft = lambda x: jnp.fft.ifft(x, axis=0)

    D = jit(lambda y,H: ifft(fft(y) * H))
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(steps):
        y = D(y, H[i])
        y = N(y, c[i])

    return y


def foe_mpowfftmax(x, M=4, sr=2*jnp.pi):
    X4 = jnp.fft.fft(x**M, axis=0)
    h  = jnp.abs(X4)**2
    f  = jnp.argmax(h, axis=0)
    N  = len(h)

    f = jnp.where(f >= N / 2, f - N, f)

    fo_hat = f / N * sr / 4

    return fo_hat, h


def measure_cd(x, sr, start=-0.25, end=0.25, bins=2000, wavlen=1550e-9):
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
    _, L = xop.scan(f, None, p)

    B2z = jnp.tan(jnp.pi/2 - (p - 1) / 2 * jnp.pi)/(sr * 2 * jnp.pi / N * sr)
    Dz_set  = -B2z / wavlen**2 * 2 * jnp.pi * c # the swept set of CD metrics
    Dz_hat = Dz_set[jnp.argmin(L)] # estimated accumulated CD

    return Dz_hat, L, Dz_set


def getpower(x):
    return jnp.mean(jnp.abs(x)**2, axis=0)


def cpe(y, x, testing_phases=4, device=gpus[0]):

    y = device_put(y, device)
    x = device_put(x, device)

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

    _, ret = xop.scan(f, None, (Y, X))

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
        fo_hat, _ = foe_mpowfftmax(y)
        return carray, fo_hat

    _, fo_hat = xop.scan(foe, None, Y)

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

    _, power = xop.scan(lambda c, y: (c, jnp.mean(jnp.abs(y)**2, axis=0)), None, yf)

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



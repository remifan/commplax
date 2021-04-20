import jax
from jax import jit, vmap, numpy as jnp, device_put
from jax.core import Value
from commplax import xop, comm, experimental as exp
from functools import partial

cpus = jax.devices('cpu')


def getpower(x, real=False):
    ''' get signal power '''
    return jnp.mean(x.real**2, axis=0) + jnp.array(1j) * jnp.mean(x.imag**2, axis=0) \
        if real else jnp.mean(abs(x)**2, axis=0)


def normpower(x, real=False):
    ''' normalize signal power '''
    if real:
        p = getpower(x, real=True)
        return x.real / jnp.sqrt(p.real) + jnp.array(1j) * x.imag / jnp.sqrt(p.imag)
    else:
        return x / jnp.sqrt(getpower(x))


def qamscale(modformat):
    M = comm.parseqamorder(modformat)
    return jnp.sqrt((M-1) * 2 / 3)


def dbp_params(
    sample_rate,                                      # sample rate of target signal [Hz]
    span_length,                                      # length of each fiber span [m]
    spans,                                            # number of fiber spans
    freqs,                                            # resulting size of linear operator
    launch_power=30,                                  # launch power [dBm]
    steps_per_span=1,                                 # steps per span
    virtual_spans=None,                               # number of virtual spans
    carrier_frequency=299792458/1550E-9,              # carrier frequency [Hz]
    fiber_dispersion=16.5E-6,                         # [s/m^2]
    fiber_dispersion_slope=0.08e3,                    # [s/m^3]
    fiber_loss=.2E-3,                                 # loss of fiber [dB]
    fiber_core_area=80E-12,                           # effective area of fiber [m^2]
    fiber_nonlinear_index=2.6E-20,                    # nonlinear index [m^2/W]
    fiber_reference_frequency=299792458/1550E-9,      # fiber reference frequency [Hz]
    ignore_beta3=False,
    polmux=True):

    # short names
    pi  = jnp.pi
    log = jnp.log
    exp = jnp.exp
    ifft = jnp.fft.ifft

    # virtual span is used in cases where we do not use physical span settings
    if virtual_spans is None:
        virtual_spans = spans

    C       = 299792458. # speed of light [m/s]
    lambda_ = C / fiber_reference_frequency
    B_2     = -fiber_dispersion * lambda_**2 / (2 * pi * C)
    B_3     = 0. if ignore_beta3 else \
        (fiber_dispersion_slope * lambda_**2 + 2 * fiber_dispersion * lambda_) * (lambda_ / (2 * pi * C))**2
    gamma   = 2 * pi * fiber_nonlinear_index / lambda_ / fiber_core_area
    LP      = 10.**(launch_power / 10 - 3)
    alpha   = fiber_loss / (10. / log(10.))
    L_eff   = lambda h: (1 - exp(-alpha * h)) / alpha
    NIter   = virtual_spans * steps_per_span
    delay   = (freqs - 1) // 2
    dw      = 2 * pi * (carrier_frequency - fiber_reference_frequency)
    w_res   = 2 * pi * sample_rate / freqs
    k       = jnp.arange(freqs)
    w       = jnp.where(k > delay, k - freqs, k) * w_res # ifftshifted

    H   = exp(-1j * (-B_2 / 2 * (w + dw)**2 + B_3 / 6 * (w + dw)**3) * \
                  span_length * spans / virtual_spans / steps_per_span)
    H_casual = H * exp(-1j * w * delay / sample_rate)
    h_casual = ifft(H_casual)

    phi = spans / virtual_spans * gamma * L_eff(span_length / steps_per_span) * LP * \
        exp(-alpha * span_length * (steps_per_span - jnp.arange(0, NIter) % steps_per_span-1) / steps_per_span)

    dims = 2 if polmux else 1

    H = jnp.tile(H[None, :, None], (NIter, 1, dims))
    h_casual = jnp.tile(h_casual[None, :, None], (NIter, 1, dims))
    phi = jnp.tile(phi[:, None, None], (1, dims, dims))

    return H, h_casual, phi


def mimoconv(y, h, mode='same', conv=xop.fftconvolve):
    conv = jnp.convolve
    dims = y.shape[-1]
    y = jnp.tile(y, (1, dims))
    h = jnp.reshape(h, (h.shape[0], dims * dims))
    zflat = jax.vmap(lambda a, b: conv(a, b, mode=mode), in_axes=-1, out_axes=-1)(y, h)
    z = zflat.T.reshape((dims, dims, -1)).sum(axis=1).T
    return z


def dbp_timedomain(y, h, c, mode='SAME', homosteps=True, scansteps=True, conv=xop.fftconvolve):

    y = device_put(y)
    h = device_put(h)
    c = device_put(c)

    dims = y.shape[-1]

    optpowscale = jnp.sqrt(dims)
    y /= optpowscale

    md = 'SAME' if homosteps else mode

    D = jit(vmap(lambda y, h: conv(y, h, mode=md), in_axes=1, out_axes=1))
    # D = jit(vmap(lambda y,h: xop.conv1d_lax(y, h), in_axes=1, out_axes=1)) # often too slow for long h


    if c.ndim == 3:
        if c.shape[1:] == (dims, dims):
            N = jit(lambda y, c: y * jnp.exp(1j * (abs(y)**2 @ c)))
            C = 0
        else:
            F = jit(vmap(lambda y, h: conv(y, h, mode='same'), in_axes=1, out_axes=1))
            N = lambda y, c: y * jnp.exp(1j * F(abs(y)**2, c))
            C = c.shape[1] - 1
    elif c.ndim == 4:
        # MIMO convolution
        F = jit(lambda p, c: mimoconv(p, c, mode='same', conv=conv))
        N = lambda y, c: y * jnp.exp(1j * F(abs(y)**2, c))
        C = c.shape[1] - 1
    else:
        raise ValueError('wrong c shape')

    T = h.shape[1] - 1
    K = h.shape[0]

    if homosteps and scansteps: # homogeneous steps is faster on first jitted run
    # scan not working on 'SAME' mode due to carry shape change
        y = xop.scan(lambda x, p: (N(D(x, p[0]), p[1]), 0.), y, (h, c))[0]
    else:
        steps = c.shape[0]
        for i in range(steps):
            y = D(y, h[i])
            y = N(y, c[i])

    if homosteps and mode.lower() == 'valid':
       y = y[K * (T + C) // 2: -K * (T + C) // 2]

    return y * optpowscale


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


def dimsdelay(x):
    dims = x.shape[-1]
    if dims <= 1:
        raise ValueError('input dimension must be at least 2 but given %d' % dims)
    x = device_put(x)
    return jax.vmap(xop.finddelay, in_axes=(None, -1), out_axes=0)(x[:, 0], x[:, 1:])


def repalign(y, x, skipfirst=0):
    N = y.shape[0]
    M = x.shape[0]
    offsets = -jax.vmap(xop.finddelay, in_axes=-1)(y[skipfirst:], x) + skipfirst
    rep = -(-N // M)
    xrep = jnp.tile(x, [rep, 1])
    z = jax.vmap(jnp.roll, in_axes=-1, out_axes=-1)(xrep, offsets)[:N, :]
    return z


def alignphase(y, x, testing_phases=4):

    y = device_put(y)
    x = device_put(x)

    TPS = jnp.exp(1j * 2 * jnp.pi / testing_phases * jnp.arange(testing_phases))

    def searchphase(y, x):
        y_t = jnp.outer(y, TPS)
        x_t = jnp.tile(x[:,None], (1, testing_phases))
        snr_t = 10. * jnp.log10(getpower(y_t) / getpower(y_t - x_t))
        return TPS[jnp.argmax(snr_t)]

    return y * vmap(searchphase, in_axes=-1)(y, x)


def localfoe(signal, frame_size=16384, frame_step=5000, sps=1, fitkind=None, degree=2,
             method=lambda x: foe_mpowfftmax(x)[0]):
    '''
    resolution = samplerate / N / 4 / sps (linear interp.)
    '''
    y = device_put(signal, cpus[0])
    dims = y.shape[-1]
    fo_local = xop.framescaninterp(y, method, frame_size, frame_step, sps)
    if fitkind is not None:
        if fitkind.lower() == 'poly':
            fo_T = jnp.tile(jnp.arange(fo_local.shape[0])[:, None], (1, dims))
            fo_local = exp.polyfitval(fo_T, fo_local, degree)
        else:
            raise ValueError('invlaid fitting method')
    return fo_local


def localpower(signal, frame_size=5000, frame_step=1000):
    y = device_put(signal)
    poweval = lambda y: jnp.mean(jnp.abs(y)**2, axis=0)
    return xop.framescaninterp(y, poweval, frame_size, frame_step)


def localdc(signal, frame_size=5000, frame_step=1000):
    y = device_put(signal)
    dceval = lambda y: jnp.mean(y, axis=0)
    return xop.framescaninterp(y, dceval, frame_size, frame_step)



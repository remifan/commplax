import numpy as np
from scipy import signal


def resample(x, p, q, axis=0):
    gcd = np.gcd(p, q)
    return signal.resample_poly(x, p//gcd, q//gcd, axis=axis)


def getpower(x, real=False):
    ''' get signal power '''
    if real:
        return np.mean(x.real**2, axis=0), np.mean(x.imag**2, axis=0)
    else:
        return np.mean(abs(x)**2, axis=0)


def normpower(x, real=False):
    ''' normalize signal power '''
    if real:
        pr, pi = getpower(x, real=True)
        return x.real / np.sqrt(pr) + 1j * x.imag / np.sqrt(pi)
    else:
        return x / np.sqrt(getpower(x))


def dbp_params(
    sample_rate,                                      # sample rate of target signal [Hz]
    span_length,                                      # length of each fiber span [m]
    spans,                                            # number of fiber spans
    freqs,                                            # resulting size of linear operator
    launch_power,                                     # launch power [dBm]
    steps_per_span=1,                                 # steps per span
    virtual_spans=None,                               # number of virtual spans
    carrier_frequency=299792458/1550.12E-9,           # carrier frequency [Hz]
    fiber_dispersion=16.7E-6,                         # [s/m^2]
    fiber_dispersion_slope=0.08e3,                    # [s/m^3]
    fiber_loss=.2E-3,                                 # loss of fiber [dB]
    fiber_core_area=80E-12,                           # effective area of fiber [m^2]
    fiber_nonlinear_index=2.6E-20,                    # nonlinear index [m^2/W]
    fiber_reference_frequency=299792458/1550.12E-9,   # fiber reference frequency [Hz]
    ignore_beta3=False,
    step_method="uniform"):

    # short names
    pi  = np.pi
    log = np.log
    exp = np.exp
    ifft = np.fft.ifft

    # virtual span is used in cases where we do not use physical span settings
    if virtual_spans is None:
        virtual_spans = spans

    C       = 299792458. # speed of light [m/s]
    lambda_ = C / fiber_reference_frequency
    B_2     = -fiber_dispersion * lambda_**2 / (2 * pi * C)
    B_3     = 0. if ignore_beta3 else \
        (fiber_dispersion_slope * lambda_**2 + 2 * fiber_dispersion * lambda_) * (lambda_ / (2 * pi * C))**2
    gamma   = 2 * pi * fiber_nonlinear_index / lambda_ / fiber_core_area
    LP      = 10.**(launch_power / 10-3)
    alpha   = fiber_loss / (10. / log(10.))
    L_eff   = lambda h: (1 - exp(-alpha * h)) / alpha
    NIter   = virtual_spans * steps_per_span
    delay   = (freqs - 1) // 2
    dw      = 2 * pi * (carrier_frequency - fiber_reference_frequency)
    w_res   = 2 * pi * sample_rate / freqs
    k       = np.arange(freqs)
    w       = np.where(k > delay, k - freqs, k) * w_res # ifftshifted

    if step_method.lower() == "uniform":
        H   = exp(-1j * (-B_2 / 2 * (w + dw)**2 + B_3 / 6 * (w + dw)**3) * \
                      span_length * spans / virtual_spans / steps_per_span)
        H_casual = H * exp(-1j * w * delay / sample_rate)
        h_casual = ifft(H_casual)

        phi = spans / virtual_spans * gamma * L_eff(span_length / steps_per_span) * LP * \
            exp(-alpha * span_length * (steps_per_span - np.arange(0, NIter) % steps_per_span-1) / steps_per_span)
    else:
        raise ValueError("step method '%s' not implemented" % step_method)

    return H, h_casual, phi


def align_periodic(y, x, begin=0, last=2000, b=0.5):

    z = np.zeros_like(x)

    def step(v, u):
        c = abs(np.correlate(u, v[begin:begin+last], "full"))
        c /= np.max(c)
        k = np.arange(-len(x)+1, len(y))
        #n = k[np.argmax(c)]

        i = np.where(c > b)[0]
        i = i[np.argsort(np.atleast_1d(c[i]))[::-1]]
        j = -k[i] + begin + last

        return j

    r0 = step(y[:,0], x[:,0])

    if len(r0) == 1: # PDM
        r0 = r0[0]
        r1 = step(y[:,1], x[:,1])[0]
    elif len(r0) == 2: # PDM Emu. ?
        r1 = r0[1]
        r0 = r0[0]
    else:
        raise RuntimeError('bad input')

    z[:,0] = np.roll(x[:,0], r0)
    z[:,1] = np.roll(x[:,1], r1)

    z = np.tile(z, (len(y)//len(z)+1,1))[:len(y),:]

    return z, np.stack((r0, r1))



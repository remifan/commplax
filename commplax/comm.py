import re
import numpy as np
import pandas as pd
from scipy import signal, special
from commplax import op
import matplotlib.pyplot as plt
import quantumrandom


def randpam(s, n, p=None):
    a = np.linspace(-s+1, s-1, s)

    return np.random.choice(a, n, p=p) + 1j * np.random.choice(a, n, p=p)


def randqam(s, n, p=None):
    m = np.int(np.sqrt(s))
    a = np.linspace(-m+1, m-1, m, dtype=np.float64)

    return np.random.choice(a, n, p=p) + 1j * np.random.choice(a, n, p=p)


def grayenc_int(x):
    x = np.asarray(x, dtype=int)

    return x ^ (x >> 1)


def graydec_int(x):
    x = np.atleast_1d(np.asarray(x, dtype=int))

    mask = np.array(x)

    while mask.any():
        I       = mask > 0
        mask[I] >>= 1
        x[I]    ^= mask[I]

    return x


def qamgrayenc_int(x, L):
    """
    Wesel, R.D., Liu, X., Cioffi, J.M. and Komninakis, C., 2001.
    Constellation labeling for linear encoders. IEEE Transactions
    on Information Theory, 47(6), pp.2417-2431.
    """
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    B = int(np.log2(M))

    x1 = x // M
    x2 = x %  M

    return (grayenc_int(x1) << B) + grayenc_int(x2)


def qamgraydec_int(x, L):
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    B = int(np.log2(M))

    x1 = graydec_int(x >> B)
    x2 = graydec_int(x % (1 << B))

    return x1 * M + x2


def pamdecision(x, L):
    x = np.asarray(x)
    y = np.atleast_1d((np.round(x / 2 + 0.5) - 0.5) * 2).astype(int)

    # apply bounds
    bd = L - 1
    y[y >  bd] =  bd
    y[y < -bd] = -bd

    return y


def qamdecision(x, L):
    x = np.atleast_1d(x)
    M = int(np.sqrt(L))

    if any(np.iscomplex(x)):
        I = pamdecision(np.real(x), M)
        Q = pamdecision(np.imag(x), M)

        y = I + 1j*Q
    else: # is tuple
        I = pamdecision(x[0], M)
        Q = pamdecision(x[1], M)
        y = (I, Q)

    return y


def qammod(x, L):
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))

    A = np.linspace(-M+1, M-1, M, dtype=np.float64)
    C = A[None,:] + 1j*A[::-1, None]

    d = qamgraydec_int(x, L)

    return C[d//M, d%M]


def qamdemod(x, L):
    x = np.asarray(x)
    M = int(np.sqrt(L))

    x = qamdecision(x, L)

    c = ((np.real(x) + M - 1) // 2).astype(int)
    r = ((M - 1 - np.imag(x)) // 2).astype(int)

    d = qamgrayenc_int(r * M + c, L)

    return d


def int2bit(d, M):
    M = np.asarray(M, dtype=np.int)
    d = np.atleast_1d(d).astype(np.uint8)
    b = np.unpackbits(d[:,None], axis=1)[:,-M:]

    return b


def bit2int(b, M):
    b = np.asarray(b, dtype=np.uint8)
    d = np.packbits(np.pad(b.reshape((-1,M)), ((0,0),(8-M,0))))

    return d


def grayqamplot(L):
    M = int(np.sqrt(L))
    x = range(L)
    y = qammod(x, L)
    fstr = "{:0" + str(M) + "b}"

    I = np.real(y)
    Q = np.imag(y)

    plt.figure(num=None, figsize=(8, 6), dpi=100)
    plt.axis('equal')
    plt.scatter(I, Q, s=1)

    for i in range(L):
        plt.annotate(fstr.format(x[i]), (I[i], Q[i]))


def const(type_str, norm=False):
    ''' generate constellation given its natrual names '''

    M = int(re.findall(r'\d+', type_str)[0])
    T = re.findall(r'[a-zA-Z]+', type_str)[0].lower()

    if T == "qam":
        C = qammod(range(M), M)
    else:
        raise ValueError('{} is not implemented yet'.format(T))

    if norm:
        C /= np.sqrt(2*(M-1)/3)

    return C


def anuqrng_bit(L):
    ''' https://github.com/lmacken/quantumrandom '''
    L    = int(L)
    N    = 0
    bits = []

    while N < L:
        b = np.unpackbits(np.frombuffer(quantumrandom.binary(), dtype=np.uint8))
        N += len(b)
        bits.append(b)

    bits = np.concatenate(bits)[:L]

    return bits


def rcosdesign(beta, span, sps, shape='normal', dtype=np.float):
    ''' ref:
        [1] https://en.wikipedia.org/wiki/Root-raised-cosine_filter
        [2] https://en.wikipedia.org/wiki/Raised-cosine_filter
        [3] Matlab R2019b rcosdesign.m
    '''

    delay = span * sps / 2
    t = np.arange(-delay, delay + 1, dtype=dtype) / sps
    b = np.zeros_like(t)
    eps = np.finfo(dtype).eps

    if beta == 0:
        beta = np.finfo(dtype).tiny

    if shape == 'normal':
        denom = 1 - (2 * beta * t) ** 2

        ind1 = np.where(abs(denom) >  np.sqrt(eps), True, False)
        ind2 = ~ind1

        b[ind1] = np.sinc(t[ind1]) * (np.cos(np.pi * beta * t[ind1]) / denom[ind1]) / sps
        b[ind2] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)

    elif shape == 'sqrt':
        ind1 = np.where(t == 0, True, False)
        ind2 = np.where(abs(abs(4 * beta * t) - 1.0) < np.sqrt(eps), True, False)
        ind3 = ~(ind1 | ind2)

        b[ind1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        b[ind2] = (
            1 / (2 * np.pi * sps)
            * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
            - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
            + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))
        )
        b[ind3] = (
            -4 * beta / sps * (np.cos((1 + beta) * np.pi * t[ind3]) +
                               np.sin((1 - beta) * np.pi * t[ind3]) / (4 * beta * t[ind3]))
            / (np.pi * ((4 * beta * t[ind3])**2 - 1))
        )

    else:
        raise ValueError('invalid shape')

    b /= np.sqrt(np.sum(b**2)) # normalize filter gain

    return b


def resample(x, p, q, axis=0):
    gcd = np.gcd(p, q)
    return signal.resample_poly(x, p//gcd, q//gcd, axis=axis)


def shape_signal(x):
    x = np.atleast_1d(np.asarray(x))

    if x.ndim == 1:
        x = x[..., None]

    return x

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
    carrier_frequency=299792458/1550E-9,           # carrier frequency [Hz]
    fiber_dispersion=16.5E-6,                         # [s/m^2]
    fiber_dispersion_slope=0.08e3,                    # [s/m^3]
    fiber_loss=.2E-3,                                 # loss of fiber [dB]
    fiber_core_area=80E-12,                           # effective area of fiber [m^2]
    fiber_nonlinear_index=2.6E-20,                    # nonlinear index [m^2/W]
    fiber_reference_frequency=299792458/1550E-9,   # fiber reference frequency [Hz]
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


def align_periodic(y, x, begin=0, last=1000, b=0.5):

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


def qamqot(y, x, count_dim=True, count_total=True, L=None):

    y = shape_signal(y)
    x = shape_signal(x)

    if L is None:
        L = len(np.unique(x))

    D = y.shape[-1]

    z = [(a, b) for a, b in zip(y.T, x.T)]

    SNR_fn = lambda y, x: 10. * np.log10(getpower(x, False) / getpower(x - y, False))

    def f(z):
        y, x = z

        M = np.sqrt(L)

        by = int2bit(qamdemod(y, L), M).ravel()
        bx = int2bit(qamdemod(x, L), M).ravel()

        BER = np.count_nonzero(by - bx) / len(by)
        with np.errstate(divide='ignore'):
            QSq = 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * BER), 0.))
        SNR = SNR_fn(y, x)

        return BER, QSq, SNR

    qot = []
    ind = []
    df = None

    if count_dim:
        qot += list(map(f, z))
        ind += ['dim' + str(n) for n in range(D)]

    if count_total:
        qot += [f((y.ravel(), x.ravel()))]
        ind += ['total']

    if len(qot) > 0:
        df = pd.DataFrame(qot, columns=['BER', 'QSq', 'SNR'], index=ind)

    return df


def qamqot_local(y, x, frame_size=10000, L=None):

    y = shape_signal(y)
    x = shape_signal(x)

    if L is None:
        L = len(np.unique(x))

    Y = op.frame(y, frame_size, frame_size, True)
    X = op.frame(x, frame_size, frame_size, True)

    zf = [(yf, xf) for yf, xf in zip(Y, X)]

    f = lambda z: qamqot(z[0], z[1], count_dim=True, L=L).to_numpy()

    qot_local = np.stack(list(map(f, zf)))

    qot_local_ip = np.repeat(qot_local, frame_size, axis=0) # better interp method?

    return {'BER': qot_local_ip[...,0], 'QSq': qot_local_ip[...,1], 'SNR': qot_local_ip[...,2]}


def corr_local(y, x, frame_size=10000, L=None):

    y = shape_signal(y)
    x = shape_signal(x)

    if L is None:
        L = len(np.unique(x))

    Y = op.frame(y, frame_size, frame_size, True)
    X = op.frame(x, frame_size, frame_size, True)

    zf = [(yf, xf) for yf, xf in zip(Y, X)]

    f = lambda z: np.abs(np.sum(z[0] * z[1].conj(), axis=0))

    qot_local = np.stack(list(map(f, zf)))

    qot_local_ip = np.repeat(qot_local, frame_size, axis=0) # better interp method?

    return qot_local_ip



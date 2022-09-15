# Copyright 2021 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import numpy as np
import pandas as pd
from scipy import signal, special
from commplax import op
import matplotlib.pyplot as plt
import quantumrandom


quasigray_32xqam = np.array([
    -3.+5.j, -1.+5.j, -3.-5.j, -1.-5.j, -5.+3.j, -5.+1.j, -5.-3.j,
    -5.-1.j, -1.+3.j, -1.+1.j, -1.-3.j, -1.-1.j, -3.+3.j, -3.+1.j,
    -3.-3.j, -3.-1.j,  3.+5.j,  1.+5.j,  3.-5.j,  1.-5.j,  5.+3.j,
    5.+1.j,  5.-3.j,  5.-1.j,  1.+3.j,  1.+1.j,  1.-3.j,  1.-1.j,
    3.+3.j,  3.+1.j,  3.-3.j,  3.-1.j
])


def qammod(x, L):
    if is_square_qam(L):
        y = square_qam_mod(x, L)
    else:
        y = cross_qam_mod(x, L)
    return y


def qamdecision(x, L):
    if is_square_qam(L):
        y = square_qam_decision(x, L)
    else:
        y = cross_qam_decision(x, L)
    return y


def qamdemod(x, L):
    if is_square_qam(L):
        d = square_qam_demod(x, L)
    else:
        d = cross_qam_decision(x, L, return_int=True)
    return d


def cross_qam_decision(x, L, return_int=False):
    x = np.asarray(x)
    c = const(L)
    idx = np.argmin(np.abs(x[:, None] - c[None, :])**2, axis=1)
    y = c[idx]
    return idx if return_int else y


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


def is_square_qam(L):
    return is_power_of_two(L) and int(np.log2(L)) % 2 == 0


def is_cross_qam(L):
    return is_power_of_two(L) and int(np.log2(L)) % 2 == 1


def cross_qam_mod(x, L):
    if L == 32:
        return quasigray_32xqam[x]
    else:
        raise ValueError(f'Cross QAM size{L} is not implemented')


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


def square_qam_grayenc_int(x, L):
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


def square_qam_graydec_int(x, L):
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


def square_qam_decision(x, L):
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


def square_qam_mod(x, L):
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    A = np.linspace(-M+1, M-1, M, dtype=np.float64)
    C = A[None,:] + 1j*A[::-1, None]
    d = square_qam_graydec_int(x, L)
    return C[d // M, d % M]


def square_qam_demod(x, L):
    x = np.asarray(x)
    M = int(np.sqrt(L))
    x = square_qam_decision(x, L)
    c = ((np.real(x) + M - 1) // 2).astype(int)
    r = ((M - 1 - np.imag(x)) // 2).astype(int)
    d = square_qam_grayenc_int(r * M + c, L)
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
    M = int(np.log2(L))
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


def parseqamorder(type_str):
    if type_str.lower() == 'qpsk':
        type_str = '4QAM'
    M = int(re.findall(r'\d+', type_str)[0])
    T = re.findall(r'[a-zA-Z]+', type_str)[0].lower()
    if T != 'qam':
        raise ValueError('{} is not implemented yet'.format(T))
    return M


def const(type_str=None, norm=False):
    ''' generate constellation given its natrual names '''

    if isinstance(type_str, str):
        M = parseqamorder(type_str)
    else:
        M = type_str
    C = qammod(range(M), M)
    if norm:
        C /= np.sqrt(2*(M-1)/3)
    return C


def canonical_qam_scale(M):
    if isinstance(M, str):
        M = parseqamorder(M)
    return np.sqrt((M-1) * 2 / 3)


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


def rcosdesign(beta, span, sps, shape='normal', dtype=np.float64):
    ''' ref:
        [1] https://en.wikipedia.org/wiki/Root-raised-cosine_filter
        [2] https://en.wikipedia.org/wiki/Raised-cosine_filter
        [3] Matlab R2019b `rcosdesign`
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


def upsample(x, n, axis=0, trim=False):
    x = np.atleast_1d(x)
    x = signal.upfirdn([1], x, n, axis=axis)
    pads = np.zeros((x.ndim, 2), dtype=int)
    pads[axis, 1] = n - 1
    y = x if trim else np.pad(x, pads)
    return y


def resample(x, p, q, axis=0):
    p = int(p)
    q = int(q)
    gcd = np.gcd(p, q)
    return signal.resample_poly(x, p//gcd, q//gcd, axis=axis)


def shape_signal(x):
    x = np.atleast_1d(np.asarray(x))

    if x.ndim == 1:
        x = x[..., None]

    return x


def getpower(x, real=False):
    ''' get signal power '''
    return np.mean(x.real**2, axis=0) + np.array(1j) * np.mean(x.imag**2, axis=0) \
        if real else np.mean(abs(x)**2, axis=0)


def normpower(x, real=False):
    ''' normalize signal power '''
    if real:
        p = getpower(x, real=True)
        return x.real / np.sqrt(p.real) + 1j * x.imag / np.sqrt(p.imag)
    else:
        return x / np.sqrt(getpower(x))


def delta(taps, dims=None, dtype=np.complex64):
    mf = np.zeros(taps, dtype=dtype)
    mf[(taps - 1) // 2] = 1.
    return mf if dims is None else np.tile(mf[:, None], dims)


def gauss(bw, taps=None, oddtaps=True, dtype=np.float64):
    """ https://en.wikipedia.org/wiki/Gaussian_filter """
    eps = 1e-8 # stablize to work with gauss_minbw
    gamma = 1 / (2 * np.pi * bw * 1.17741)
    mintaps = int(np.ceil(6 * gamma - 1 - eps))
    if taps is None:
        taps = mintaps
    elif taps < mintaps:
        raise ValueError('required {} taps which is less than minimal default {}'.format(taps, mintaps))

    if oddtaps is not None:
        if oddtaps:
            taps = mintaps if mintaps % 2 == 1 else mintaps + 1
        else:
            taps = mintaps if mintaps % 2 == 0 else mintaps + 1
    return gauss_kernel(taps, gamma, dtype=dtype)


def gauss_minbw(taps):
    return 1 / (2 * np.pi * ((taps + 1) / 6) * 1.17741)


def gauss_kernel(n=11, sigma=1, dims=None, dtype=np.complex64):
    r = np.arange(-int(n / 2), int(n / 2) + 1) if n % 2 else np.linspace(-int(n / 2) + 0.5, int(n / 2) - 0.5, n)
    w = np.array([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x)**2 / (2 * sigma**2)) for x in r]).astype(dtype)
    return w if dims is None else np.tile(w[:, None], dims)


def qamscale(modformat):
    if isinstance(modformat, str):
        M = parseqamorder(modformat)
    else:
        M = modformat
    return np.sqrt((M - 1) * 2 / 3) if is_square_qam(M) else np.sqrt(2/3 * (M * 31/32 - 1))


def dbp_params(
    sample_rate,                                      # sample rate of target signal [Hz]
    span_length,                                      # length of each fiber span [m]
    spans,                                            # number of fiber spans
    freqs,                                            # resulting size of linear operator
    launch_power=0,                                   # launch power [dBm]
    steps_per_span=1,                                 # steps per span
    virtual_spans=None,                               # number of virtual spans
    carrier_frequency=194.1e12,                       # carrier frequency [Hz]
    fiber_dispersion=16.7E-6,                         # [s/m^2]
    fiber_dispersion_slope=0.08e3,                    # [s/m^3]
    fiber_loss=.2E-3,                                 # loss of fiber [dB]
    fiber_core_area=80E-12,                           # effective area of fiber [m^2]
    fiber_nonlinear_index=2.6E-20,                    # nonlinear index [m^2/W]
    fiber_reference_frequency=194.1e12,      # fiber reference frequency [Hz]
    ignore_beta3=False,
    polmux=True,
    domain='time',
    step_method="uniform"):

    domain = domain.lower()
    assert domain == 'time' or domain == 'frequency'

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
    LP      = 10.**(launch_power / 10 - 3)
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

    if polmux:
        dims = 2
    else:
        dims = 1

    H = np.tile(H[None, :, None], (NIter, 1, dims))
    h_casual = np.tile(h_casual[None, :, None], (NIter, 1, dims))
    phi = np.tile(phi[:, None, None], (1, dims, dims))

    return (h_casual, phi) if domain == 'time' else (H, phi)


def finddelay(x, y):
    '''
    case 1:
        X = [1, 2, 3]
        Y = [0, 0, 1, 2, 3]
        D = comm.finddelay(X, Y) # D = 2
    case 2:
        X = [0, 0, 1, 2, 3, 0]
        Y = [0.02, 0.12, 1.08, 2.21, 2.95, -0.09]
        D = comm.finddelay(X, Y) # D = 0
    case 3:
        X = [0, 0, 0, 1, 2, 3, 0, 0]
        Y = [1, 2, 3, 0]
        D = comm.finddelay(X, Y) # D = -3
    case 4:
        X = [0, 1, 2, 3]
        Y = [1, 2, 3, 0, 0, 0, 0, 1, 2, 3, 0, 0]
        D = comm.finddelay(X, Y) # D = -1
    reference:
        https://www.mathworks.com/help/signal/ref/finddelay.html
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    c = abs(signal.correlate(x, y, mode='full', method='fft'))
    k = np.arange(-len(y)+1, len(x))
    i = np.lexsort((np.abs(k), -c))[0] # lexsort to handle case 4
    d = -k[i]
    return d


def align_periodic(y, x, begin=0, last=2000, b=0.5):

    dims = x.shape[-1]
    z = np.zeros_like(x)

    def step(v, u):
        c = abs(signal.correlate(u, v[begin:begin+last], mode='full', method='fft'))
        c /= np.max(c)
        k = np.arange(-len(x)+1, len(y))
        #n = k[np.argmax(c)]

        i = np.where(c > b)[0]
        i = i[np.argsort(np.atleast_1d(c[i]))[::-1]]
        j = -k[i] + begin + last

        return j

    r0 = step(y[:,0], x[:,0])

    if dims > 1:
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
        d = np.stack((r0, r1))
    else:
        z[:,0] = np.roll(x[:,0], r0)
        d = r0

    z = np.tile(z, (len(y)//len(z)+1,1))[:len(y),:]

    return z, d


def qamqot(y, x, count_dim=True, count_total=True, L=None, eval_range=(0, 0), scale=1):
    #if checktruthscale:
    #    assert L is not None
    #    ux = np.unique(x)
    #    powdiff = abs(getpower(ux) - getpower(const(str(L) + 'QAM')))
    #    if  powdiff > 1e-4:
    #        #TODO add warning colors
    #        print("truth QAM data is not properly scaled to its canonical form, scale = %.5f" % powdiff)


    assert y.shape[0] == x.shape[0]
    y = y[eval_range[0]: y.shape[0] + eval_range[1] if eval_range[1] <= 0 else eval_range[1]] * scale
    x = x[eval_range[0]: x.shape[0] + eval_range[1] if eval_range[1] <= 0 else eval_range[1]] * scale

    y = shape_signal(y)
    x = shape_signal(x)

    # check if scaled x is canonical
    p = np.rint(x.real) + 1j * np.rint(x.imag)
    if np.max(np.abs(p - x)) > 1e-2:
        raise ValueError('the scaled x is seemly not canonical')

    if L is None:
        L = len(np.unique(p))

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


def qamqot_local(y, x, frame_size=10000, L=None, scale=1, eval_range=None):

    y = shape_signal(y)
    x = shape_signal(x)

    if L is None:
        L = len(np.unique(x))

    Y = op.frame(y, frame_size, frame_size, True)
    X = op.frame(x, frame_size, frame_size, True)

    zf = [(yf, xf) for yf, xf in zip(Y, X)]

    f = lambda z: qamqot(z[0], z[1], count_dim=True, L=L, scale=scale).to_numpy()

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


def snrstat(y, x, frame_size=10000, L=None, eval_range=(0, 0), scale=1):
    assert y.shape[0] == x.shape[0]
    y = y[eval_range[0]: y.shape[0] + eval_range[1] if eval_range[1] <= 0 else eval_range[1]] * scale
    x = x[eval_range[0]: x.shape[0] + eval_range[1] if eval_range[1] <= 0 else eval_range[1]] * scale
    snr_local = qamqot_local(y, x, frame_size, L)['SNR'][:, :2]
    sl_mean = np.mean(snr_local, axis=0)
    sl_std = np.std(snr_local, axis=0)
    sl_max = np.max(snr_local, axis=0)
    sl_min = np.min(snr_local, axis=0)
    return np.stack((sl_mean, sl_std, sl_mean - sl_min, sl_max - sl_mean))


def firfreqz(h, sr=1, N=8192, t0=None, bw=None):
    if h.ndim == 1:
        h = h[None,:]

    T = h.shape[-1]

    if t0 is None:
        t0 = (T - 1) // 2 + 1

    H = []
    for hi in h:
        w, Hi = signal.freqz(hi, worN=N, whole=True)
        Hi *= np.exp(1j * w * (t0 - 1))
        H.append(Hi)
    H = np.array(H)

    w = (w + np.pi) % (2 * np.pi) - np.pi

    H = np.squeeze(np.fft.fftshift(H, axes=-1))
    w = np.fft.fftshift(w, axes=-1) * sr / 2 / np.pi

    if bw is not None:
        s = int((sr - bw) / sr / 2 * len(w))
        w = w[s: -s]
        H = H[..., s: -s]

    # w = np.fft.fftshift(np.fft.fftfreq(H.shape[-1], 1/sr))

    return w, H



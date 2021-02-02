import numpy as np
from scipy import signal
import scipy.io as spio
import matplotlib.pyplot as plt
from commplax import src
from commplax.contrib import frft, ifrft

def xcorr(y, x):
    x = np.asarray(x)
    y = np.asarray(y)

    # global xcorr
    c = abs(np.correlate(x, y, "full"))
    k = np.arange(-len(x)+1, len(y))
    n = k[np.argmax(c)]

    fig = plt.figure()
    plt.plot(k, c)
    plt.title('global xcorr, y[t{}{}] approx. x[t] most likely'.format('+' if n < 0 else '-', abs(n)))
    plt.xlabel('offset')
    plt.ylabel('abs(xcorr)')

def pownspec(y, n=4):
    dft = lambda a: np.fft.fftshift(np.fft.fft(a, axis=0), axes=0)

    fig = plt.figure(figsize=(6, 2*n))
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.plot(abs(dft(np.power(y, i + 1))))

def cdmetric(x, sr, start=-0.25, end=0.25, bins=200, wavlen=1550e-9):
    '''
    Zhou, H., Li, B., Tang, et. al, 2016. Fractional fourier transformation-based
    blind chromatic dispersion estimation for coherent optical communications.
    Journal of Lightwave Technology, 34(10), pp.2371-2380.
    '''
    c = 299792458.
    p = np.linspace(start, end, bins)
    N = len(x)
    K = len(p)

    L = np.zeros(K, dtype=np.float)
    for i in range(K):
        L[i] = np.sum(abs(
            frft(abs(frft(x, p[i]))**2, -1)
        )**2)

    B2z = np.tan(np.pi/2 - (p - 1) / 2 * np.pi)/(sr * 2 * np.pi / N * sr)
    Dz  = -B2z / wavlen**2 * 2 * np.pi * c
    Dze = Dz[np.argmin(L)] # estimated accumulated CD

    return Dze, L, Dz

def local_corr(y, x, wsize=1000):
    # local corr
    T = np.min([len(x), len(y)])
    ct = np.zeros(T)
    win = np.array([t for t in range(-(wsize-1)//2, (wsize-1)//2+1)])
    for t in range(T):
        ind = (win + t) % T
        ct[t] = abs(np.dot(x.conj()[ind], y[ind]))

    fig = plt.figure()

    plt.plot(range(T), ct)
    plt.title('local corr, winsize={}'.format(wsize))
    plt.xlabel('t')
    plt.ylabel('abs(corr)')

def snr_ema(y, x, beta=0.998):
    ''' exp. moving avg of SNR (dB) '''
    snr = []
    snr_m = 0
    eps = 1e-20 # for numberical stability
    for y_i, x_i in zip(y, x):
        snr_i = 10 * np.log10((abs(y_i) / abs(y_i - x_i)) ** 2 + eps)
        snr_m = beta * snr_m + (1 - beta) * snr_i
        snr.append(snr_m)

    plt.figure()
    plt.plot(snr)

def snr_sma(y, x, n=50):
    ''' exp. moving avg of SNR (dB) '''
    snr = []
    eps = 1e-20 # for numberical stability

    snr = 10 * np.log10( (abs(y) / abs(y - x))**2 + eps )

    if snr.ndim == 1:
        snr = snr[:, None]

    nch = snr.shape[-1]
    h   = np.ones(n, dtype=np.float)

    for ch in range(nch):
        snr[:,ch] = np.convolve(snr[:,ch], h, mode='same') / n

    plt.figure()
    plt.plot(snr)

def se_dist(y, x, L, bins=None):

    yd = src.qamdemod(y, L)
    xd = src.qamdemod(x, L)

    sed = np.where(yd != xd)

    plt.figure()
    plt.hist(sed, bins=bins)



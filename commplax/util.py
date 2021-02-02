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
        L[i] = np.sum(abs(frft(abs(frft(x, p[i]))**2, -1))**2)

    B2z = np.tan(np.pi/2 - (p - 1) / 2 * np.pi)/(sr * 2 * np.pi / N * sr)
    Dz_set  = -B2z / wavlen**2 * 2 * np.pi * c
    Dz_hat = Dz_set[np.argmin(L)] # estimated accumulated CD

    return Dz_hat, L, Dz_set

def firfreqz(h, sr=1, N=8192, t0=None):
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

    H = np.fft.fftshift(H, axes=-1)
    w = np.fft.fftshift(w, axes=-1) * sr / 2 / np.pi

    # w = np.fft.fftshift(np.fft.fftfreq(H.shape[-1], 1/sr))

    return w, np.squeeze(H)


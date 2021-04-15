import numpy as np
from scipy import signal


def dbp_2d(y, h, phi):

    niter = len(phi)

    for i in range(niter):
        y[:,0] = np.convolve(y[:,0], h[i,:,0], mode='same')
        y[:,1] = np.convolve(y[:,1], h[i,:,1], mode='same')

    return y


def resample(x, p, q, axis=0):
    gcd = np.gcd(p, q)
    return signal.resample_poly(x, p//gcd, q//gcd, axis=axis)


def frame_prepare(x, flen, fstep, pad_end=False, pad_constants=0):
    x = np.asarray(x)
    n = x.shape[0]

    if n < flen:
        raise ValueError('array length {} < frame length {}'.format(n, flen))

    if flen < fstep:
        raise ValueError('frame length {} < frame step {}'.format(flen, fstep))

    if pad_end:
        n = int(np.ceil(n))
        fnum = -(-n // fstep) # double negatives to round up
        pad_len = (fnum - 1) * fstep + flen - n
        pad_width = ((0,pad_len),) + ((0,0),) * (x.ndim-1)
        x = np.pad(x, pad_width, 'constant', constant_values=(pad_constants, pad_constants))
    else:
        fnum = 1 + (n - flen) // fstep
        n = (fnum - 1) * fstep + flen
        # Truncate to final length.
        x = x[:n,...]

    return x, fnum


def frame_gen(x, flen, fstep, pad_end=False, pad_constants=0):
    x, fnum = frame_prepare(x, flen, fstep, pad_end=pad_end, pad_constants=pad_constants)

    s = np.arange(flen)

    for i in range(fnum):
        yield x[s + i * fstep,...]


def frame(x, flen, fstep=None, pad_end=False, pad_constants=0):
    if fstep is None:
        fstep = flen

    x, fnum = frame_prepare(x, flen, fstep, pad_end=pad_end, pad_constants=pad_constants)

    ind = np.arange(flen)[None,:] + fstep * np.arange(fnum)[:,None]
    return x[ind,...]


def delay(x, d):
    return np.roll(x, d, axis=0)



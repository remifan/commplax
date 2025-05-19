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


import numpy as np
from scipy import signal
from functools import partial


def dbp_2d(y, h, phi):

    niter = len(phi)

    for i in range(niter):
        y[:,0] = np.convolve(y[:,0], h[i,:,0], mode='same')
        y[:,1] = np.convolve(y[:,1], h[i,:,1], mode='same')

    return y


def vconv(x, y, mode='full', **kwargs):
    """ vectorized convolution """
    conv = np.vectorize(partial(np.convolve, mode=mode, **kwargs), signature='(n),(m)->(k)')
    z = conv(x.T, y.T).T
    return z


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


def frame_shape(s, flen, fstep, pad_end=False, allowwaste=True):
    n = s[0]
    ndim = len(s)

    if ndim < 2:
        raise ValueError('rank must be atleast 2, got %d instead' % ndim)

    if n < flen:
        raise ValueError('array length {} < frame length {}'.format(n, flen))

    if flen < fstep:
        raise ValueError('frame length {} < frame step {}'.format(flen, fstep))

    if pad_end:
        fnum = -(-n // fstep) # double negatives to round up
        pad_len = (fnum - 1) * fstep + flen - n
        n = n + pad_len
    else:
        waste = (n - flen) % fstep
        if not allowwaste and waste != 0:
            raise ValueError('waste %d' % waste)
        fnum = 1 + (n - flen) // fstep
        n = (fnum - 1) * fstep + flen

    return (fnum, flen) + s[1:]


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



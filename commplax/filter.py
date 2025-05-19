# Copyright 2025 The Commplax Authors.
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


def rcosdesign(beta, span, sps, shape='normal', norm_gain=False, dtype=np.float64):
    ''' 
        implementation follows the descriptions of [1,2], function interface follows [3]
        ref:
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

    if norm_gain:
        b /= np.sqrt(np.sum(b**2)) # normalize filter gain

    return b


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


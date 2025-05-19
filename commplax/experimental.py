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
from jax import jit, vmap, numpy as jnp
from numpy.polynomial.polyutils import RankWarning
from functools import partial


def polyfit(x, y, deg, rcond=None):
    '''
    Reference:
    https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/polynomial/polyutils.py#L641
    [BUG]:
    1. jit on GPU somehow demands insanely massive memeory
    2. inaccurate by using default float32 based rcond, use np.float64 as workaround
    '''
    x = jnp.atleast_1d(x) + 0.0
    y = jnp.atleast_1d(y) + 0.0
    order = np.asarray(deg) + 1
    if rcond is None:
        # standard rcond is buggy in jax, might be related to float32
        # rcond = x.shape[0] * jnp.finfo(x.dtype).eps
        rcond = x.shape[0] * jnp.finfo(np.float64).eps

    c, [resids, rank, s, rcond] = jit(_polyfit, static_argnums=(2, 3), backend='cpu')(x, y, order, rcond)
    # if rank != order and not full:
    #     msg = "The fit may be poorly conditioned, rank=%d, oder=%d" % (rank, order)
    #     warnings.warn(msg, RankWarning, stacklevel=2)

    return c


def _polyfit(x, y, order, rcond):
    lhs = jnp.vander(x, order).T
    rhs = y.T
    scl = jnp.sqrt(jnp.square(lhs).sum(1))
    scl = jnp.where(scl == 0, 1, scl)
    c, resids, rank, s = jnp.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T
    return c, [resids, rank, s, rcond]


def polyfitval(x, y, deg):
    return vmap(lambda x, y: jnp.polyval(polyfit(x, y, deg), x), in_axes=-1, out_axes=-1)(x, y)


def ifwmtriplets(y, n=1, m=1):
    return _ifwmtriplets(y, n, m)


partial(jit, static_argnums=(1, 2))
def _ifwmtriplets(y, n, m):
    N = y.shape[0]
    if y.shape[-1] != 2:
        raise ValueError('only polmux signal is allowed')
    h = y[:, 0]
    v = y[:, 1]

    boundi = lambda i: jnp.where((0 <= i) & (i < N), i, 0)
    boundv = lambda i, a: jnp.where((0 <= i) & (i < N), a[i], 0)

    t = jnp.arange(N)
    m = jnp.arange(-m, m + 1)
    n = jnp.arange(-n, n + 1)
    k = m[:, None] + n[None, :]
    tm = m[None, :] + t[:, None]
    tn = n[None, :] + t[:, None]
    tk = k[None, :] + t[:, None, None]
    tm = boundi(tm)
    tn = boundi(tn)
    tk = boundi(tk)
    hm = boundv(tm, h)
    hn = boundv(tn, h)
    hk = boundv(tk, h)
    vm = boundv(tm, v)
    vn = boundv(tn, v)
    vk = boundv(tk, v)

    nlh = hm[:, :, None] * hn[:, None, :] * hk.conj() + vm[:, :, None] * hn[:, None, :] * vk.conj()
    nlv = vm[:, :, None] * vn[:, None, :] * vk.conj() + hm[:, :, None] * vn[:, None, :] * hk.conj()
    return jnp.stack([nlh, nlv], axis=-1)


# x0, t0 = scope.child(conv1d, name='Dx_%d' % i)(Signal(x[:, 0], t[:, 0]), taps=dtaps, kernel_init=d_init)
# x1, t1 = scope.child(conv1d, name='Dy_%d' % i)(Signal(x[:, 1], t[:, 1]), taps=dtaps, kernel_init=d_init)
# x = jnp.stack([x0, x1], axis=-1)
# t = jnp.stack([t0, t1], axis=-1)

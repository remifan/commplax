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


import re
import jax
import numpy as np
from jax import numpy as jnp, random as jr
from commplax import jax_util as ju


quasigray_32xqam = jnp.array([
    -3.+5.j, -1.+5.j, -3.-5.j, -1.-5.j, -5.+3.j, -5.+1.j, -5.-3.j,
    -5.-1.j, -1.+3.j, -1.+1.j, -1.-3.j, -1.-1.j, -3.+3.j, -3.+1.j,
    -3.-3.j, -3.-1.j,  3.+5.j,  1.+5.j,  3.-5.j,  1.-5.j,  5.+3.j,
    5.+1.j,  5.-3.j,  5.-1.j,  1.+3.j,  1.+1.j,  1.-3.j,  1.-1.j,
    3.+3.j,  3.+1.j,  3.-3.j,  3.-1.j
])


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


def is_square_qam(L):
    return is_power_of_two(L) and int(np.log2(L)) % 2 == 0


def is_cross_qam(L):
    return is_power_of_two(L) and int(np.log2(L)) % 2 == 1


def grayenc_int(x):
    x = jnp.asarray(x, dtype=jnp.int32)
    return x ^ (x >> 1)


def graydec_int(x):
    x = jnp.atleast_1d(jnp.asarray(x, dtype=jnp.int32))
    m = jnp.copy(x) # mask
    cond = lambda state:  state[1].any()
    def body(state):
        x, m = state
        I = m > 0
        m = jnp.where(I, m>>1, m)
        x = jnp.where(I, x^m, x)
        state = x, m
        return state
    y = jax.lax.while_loop(cond, body, (x, m))[0]
    return y


def parse_modulation(type_str):
    '''Parse modulation format string like "16QAM", "PAM4", "QPSK", etc.

    Returns:
        (order, type): tuple of (int, str) e.g. (16, 'qam') or (4, 'pam')
    '''
    type_str = type_str.strip()
    if type_str.lower() == 'qpsk':
        return 4, 'qam'
    M = int(re.findall(r'\d+', type_str)[0])
    T = re.findall(r'[a-zA-Z]+', type_str)[0].lower()
    if T not in ('qam', 'pam'):
        raise ValueError('{} is not implemented yet'.format(T))
    return M, T


# Keep for backward compatibility
def parseqamorder(type_str):
    M, T = parse_modulation(type_str)
    if T != 'qam':
        raise ValueError('{} is not a QAM format'.format(type_str))
    return M


def pamconst(M):
    '''Generate PAM-M constellation points.

    Args:
        M: PAM order (e.g., 2, 4, 8)

    Returns:
        Array of M real-valued constellation points: [-(M-1), ..., -1, 1, ..., (M-1)]
    '''
    return jnp.linspace(-M + 1, M - 1, M, dtype=ju.default_floating_dtype())


def const(type_str=None, norm=False):
    '''Generate constellation given its natural names.

    Args:
        type_str: Modulation format string, e.g., "16QAM", "PAM4", "4PAM", "QPSK"
        norm: If True, normalize to unit average power

    Returns:
        Array of constellation points (complex for QAM, real for PAM)
    '''
    if isinstance(type_str, str):
        M, T = parse_modulation(type_str)
    else:
        # Legacy: assume QAM if just a number is passed
        M, T = type_str, 'qam'

    if T == 'pam':
        C = pamconst(M)
        if norm:
            # PAM-M power = (M^2 - 1) / 3
            C = C / jnp.sqrt((M**2 - 1) / 3)
    else:  # qam
        C = qammod(range(M), M)
        if norm:
            C = C / jnp.sqrt(2 * (M - 1) / 3)
    return C


def square_qam_grayenc_int(x, L):
    """
    Wesel, R.D., Liu, X., Cioffi, J.M. and Komninakis, C., 2001.
    Constellation labeling for linear encoders. IEEE Transactions
    on Information Theory, 47(6), pp.2417-2431.
    """
    x = jnp.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    B = int(np.log2(M))
    x1 = x // M
    x2 = x %  M
    return (grayenc_int(x1) << B) + grayenc_int(x2)


def square_qam_graydec_int(x, L):
    x = jnp.asarray(x, dtype=jnp.int32)
    M = int(np.sqrt(L))
    B = int(np.log2(M))
    x1 = graydec_int(x >> B)
    x2 = graydec_int(x % (1 << B))
    return x1 * M + x2


def pamdecision(x, L):
    x = jnp.asarray(x)
    y = jnp.atleast_1d((jnp.round(x / 2 + 0.5) - 0.5) * 2).astype(jnp.int32)
    # apply bounds
    bd = L - 1
    y = jnp.where(y >  bd,  bd, y)
    y = jnp.where(y < -bd, -bd, y)
    return y


def square_qam_decision(x, L):
    x = jnp.atleast_1d(x)
    M = int(np.sqrt(L))
    if isinstance(x, tuple):
        I = pamdecision(x[0], M)
        Q = pamdecision(x[1], M)
        y = (I, Q)
    else:
        I = pamdecision(jnp.real(x), M)
        Q = pamdecision(jnp.imag(x), M)
        y = I + 1j*Q
    return y


def square_qam_mod(x, L):
    x = np.asarray(x, dtype=int)
    M = int(np.sqrt(L))
    A = jnp.linspace(-M+1, M-1, M, dtype=ju.default_floating_dtype())
    C = A[None,:] + 1j*A[::-1, None]
    d = square_qam_graydec_int(x, L)
    return C[d // M, d % M]


def cross_qam_mod(x, L):
    if L == 32:
        return quasigray_32xqam[x]
    else:
        raise ValueError(f'Cross QAM size{L} is not implemented')


def cross_qam_decision(x, L, return_int=False):
    x = jnp.asarray(x)
    c = const(L)
    idx = jnp.argmin(jnp.abs(x[:, None] - c[None, :])**2, axis=1)
    y = c[idx]
    return idx if return_int else y


def square_qam_demod(x, L):
    x = jnp.asarray(x)
    M = int(np.sqrt(L))
    x = square_qam_decision(x, L)
    c = ((jnp.real(x) + M - 1) // 2).astype(jnp.int32)
    r = ((M - 1 - jnp.imag(x)) // 2).astype(jnp.int32)
    d = square_qam_grayenc_int(r * M + c, L)
    return d


def int2bit(d, M):
    M = int(M)
    d = jnp.atleast_1d(d).astype(jnp.uint8)
    b = jnp.unpackbits(d[:, None], axis=1)[:,-M:]
    return b


def bit2int(b, M):
    M = int(M)
    b = jnp.asarray(b, dtype=jnp.uint8)
    d = jnp.packbits(jnp.pad(b.reshape((-1,M)), ((0,0),(8-M,0))))
    return d
 

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
 

def qammod(x, L):
    if is_square_qam(L):
        y = square_qam_mod(x, L)
    else:
        y = cross_qam_mod(x, L)
    return y


def randpam(key, s, n, p=None):
    n = ju.astuple(n)
    a = jnp.linspace(-s+1, s-1, s, dtype=ju.default_floating_dtype())
    return jr.choice(key, a, n, p=p)


def randqam(key, s, n, p=None):
    key_i, key_q = jr.split(key)
    n = ju.astuple(n)
    m = int(np.sqrt(s))
    a = np.linspace(-m+1, m-1, m, dtype=ju.default_floating_dtype())
    return jr.choice(key_i, a, n, p=p) + 1j * jr.choice(key_q, a, n, p=p)


def getpower(x, real=False):
    ''' get signal power '''
    return jnp.mean(x.real**2, axis=0) + jnp.array(1j) * jnp.mean(x.imag**2, axis=0) \
        if real else jnp.mean(jnp.abs(x)**2, axis=0)


def measure(y, x, L=16):
    M = np.sqrt(L)

    sy = qamdemod(y, L)
    sx = qamdemod(x, L)
    by = int2bit(sy, M).ravel()
    bx = int2bit(sx, M).ravel()

    SNR_fn = lambda y, x: 10. * jnp.log10(getpower(x, False) / getpower(x - y, False))
    SNRdB2EVM = lambda x: 1 / jnp.sqrt(jnp.power(10, x/10))

    ber = jnp.count_nonzero(by - bx) / by.shape[0]
    ser = jnp.count_nonzero(sy - sx) / len(sy)
    snr = SNR_fn(y, x)
    evm = SNRdB2EVM(snr)
    return {'BER': ber, 'SER': ser, 'SNR': snr, 'EVM': evm}

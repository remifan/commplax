import re
import jax
import jax.numpy as jnp
import numpy as np


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
    mask = jnp.asarray(x)
    cond = lambda mask:  mask.any()
    def body(mask):
        I    = mask > 0
        mask = jnp.where(I, mask>>1, mask)
        x    = jnp.where(I, x ^ mask, x)
        return x
    x = jax.lax.while_loop(cond, body, mask)
    return x


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
        C = C / jnp.sqrt(2*(M-1)/3)
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
    A = jnp.linspace(-M+1, M-1, M, dtype=jnp.float64)
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

import re
import numpy as np
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



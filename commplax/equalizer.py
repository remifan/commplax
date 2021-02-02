import numpy as np
import jax
from jax import lax, jit, vmap, numpy as jnp, device_put
from jax.ops import index, index_add, index_update
from commplax import src

def tddbp_params(
    SR,                        # sample rate
    LSpan,                     # length of each fiber span
    NSpan,                     # number of fiber spans
    VSpan,                     # number of virtual spans
    NTaps,                     # tap number of CD-comp. filter
    LP_dBm,                    # launch power
    StPS=1,                    # steps per span
    D=-16.7E-6,                # CD coef.
    Loss_dB=.2E-3,             # loss of fiber
    A_eff=80E-12,              # effective area of fiber
    N_2=2.6E-20,               # nonlinear index
    fc=299792458/1550.12E-9,   # reference frequency
    cxdt=np.complex64,         # dtype of complex values
    fltdt=np.float32,          # dtype of real values
    np=np):                    # numpy interfaces backend

    # toflt = lambda X: tuple(map(lambda x: np.asarray(x, dtype=fltdt), X))
    # tocx  = lambda X: tuple(map(lambda x: np.asarray(x, dtype=cxdt), X))

    pi        = np.pi
    log       = np.log
    exp       = np.exp
    ifftshift = np.fft.ifftshift
    ifft      = np.fft.ifft

    c     = 299792458.
    Lamda = c/fc
    B_2   = -D*Lamda**2/(2*pi*c)
    Gamma = 2*pi*N_2/Lamda/A_eff
    LP    = 10.**(LP_dBm/10-3)
    Alpha = Loss_dB/(10./log(10.))
    L_eff = lambda h: (1-exp(-Alpha*h))/Alpha
    w_res = 2*pi*SR/NTaps
    w     = np.arange(-SR*pi, SR*pi, w_res)
    H     = ifftshift(exp(-1j*B_2/2*w**2*LSpan/StPS*NSpan/VSpan) * \
                      exp(-1j*w*((NTaps-1)//2/SR)))
    NIter = VSpan * StPS
    h     = np.tile(ifft(H), (NIter,1))
    phi   = NSpan / VSpan * Gamma * L_eff(LSpan/StPS) * LP * \
        exp(-Alpha*LSpan*(StPS-np.arange(0,NIter)%StPS-1)/StPS)

    return h, phi


def conv1d_lax(x, h):
    '''
    damn slow in CPU, jaxlib-cuda (cudnn's impl.) is recommended
    '''
    x = device_put(x)
    h = device_put(h)

    x = x[jnp.newaxis,:,jnp.newaxis]
    h = h[::-1,jnp.newaxis,jnp.newaxis]
    dn = lax.conv_dimension_numbers(x.shape, h.shape, ('NWC', 'WIO', 'NWC'))

    # lax.conv_general_dilated runs much slower than numpy.convolve on CPU_device
    x = lax.conv_general_dilated(x,      # lhs = image tensor
                                 h,      # rhs = conv kernel tensor
                                 (1,),   # window strides
                                 'SAME', # padding mode
                                 (1,),   # lhs/image dilation
                                 (1,),   # rhs/kernel dilation
                                 dn)     # dimension_numbers = lhs, rhs, out dimension permu

    return x[0,:,0]


def dbp_2d_lax(y, h, c):
    niter = len(c)

    y = device_put(y)
    h = device_put(h)
    c = device_put(c)

    D = jit(vmap(lambda y,h: conv1d_lax(y, h), in_axes=1, out_axes=1))
    N = jit(lambda y,c: y * jnp.exp(1j * (abs(y)**2 @ c)))

    for i in range(niter):
        y = D(y, h[i])
        y = N(y, c[i])

    return y


def dbp_2d(y, h, phi):
    niter = len(phi)

    for _ in range(niter):
        y[:,0] = np.convolve(y[:,0], h, mode='same')
        y[:,1] = np.convolve(y[:,1], h, mode='same')

    return y


def foe_4s(x, sr=2*np.pi, dft=lambda a: np.fft.fft(a, axis=0)):
    X4 = dft(x**4)
    h  = abs(X4)**2
    f  = np.argmax(h, axis=0)
    N  = len(h)

    f = np.where(f >= N / 2, f - N, f)

    fo_hat = f / N * sr / 4

    return fo_hat, h


def mimo(w, u, np=jnp):
    v = np.einsum('ijt,tj->i', w, u)

    return v


def loss_cma(v, const=src.const("16QAM", norm=True), np=jnp):
    d = np.mean(abs(const)**4) / np.mean(abs(const)**2)
    l = np.sum(np.abs(d - np.abs(v)**2))

    return l


def grad_cma(w, u, v, const=src.const("16QAM", norm=True), np=jnp):
    d = np.mean(abs(const)**4) / np.mean(abs(const)**2)
    return -2 * (v * (d - abs(v)**2))[...,None,None] * np.conj(u).T[None,...]


def resample(x, p, q, **kwargs):
    return signal.resample_poly(x, p, q, axis=0)


def getpower(x, real=False):
    ''' get signal power '''
    if real:
        return np.mean(x.real**2, axis=0), np.mean(x.imag**2, axis=0)
    else:
        return np.mean(abs(x)**2, axis=0)


def normpower(x, real=False):
    ''' normalize signal power '''
    if real:
        pr, pi = getpower(x, real=True)
        return x.real / np.sqrt(pr) + 1j * x.imag / np.sqrt(pi)
    else:
        return x / np.sqrt(getpower(x))


def frame(x, flen, fstep, pad_end=False, pad_value=0, np=jnp):
    x = np.array(x)
    N = len(x)

    if N <= flen:
        raise ValueError('input length <= frame length')

    if flen < fstep:
        raise ValueError('frame length < frame step')

    n = (N - flen) / fstep + 1 # number of frames

    # pad x otherwise clip
    if pad_end:
        n = int(np.ceil(n))
        pad_len = (n - 1) * fstep + flen - N
        pad_width = ((0,pad_len),) + ((0,0),) * (x.ndim-1)
        x = np.pad(x, pad_width)
        N = len(x)
    else:
        n = np.floor(n).astype(np.int32)
        N = (n - 1) * fstep + flen
        x = x[:N,...]

    i = np.arange(flen)[None,:] + fstep * np.arange(n)[:,None]
    y = x[i,...]

    return y



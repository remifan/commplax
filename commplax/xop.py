import numpy as np
from jax import lax, jit, vmap, numpy as jnp, device_put
from jax.ops import index, index_add, index_update
from functools import partial


def isfloat(x):
    return issubclass(x.dtype.type, np.floating)


def iscomplex(x):
    return issubclass(x.dtype.type, np.complexfloating)


def scan(f, init, xs, length=None, reverse=False, unroll=1, jit_device=None, jit_backend=None):
    '''
    "BUG: ``lax.scan`` is known to cause memory leaks when not called within a jitted function"
    "https://github.com/google/jax/issues/3158#issuecomment-631851006"
    "https://github.com/google/jax/pull/5029/commits/977c9c40efa378d1321a7dd8c712af528939ed5f"
    "https://github.com/google/jax/pull/5029"
    "NOTE": ``scan`` runs much slower on GPU than CPU if loop iterations are small (GPU IO bottleneck?)
    "https://github.com/google/jax/issues/2491"
    "https://github.com/google/jax/pull/3076"
    '''
    @partial(jit, static_argnums=(0,3,4,5), device=jit_device, backend=jit_backend)
    def _scan(f, init, xs, length, reverse, unroll):
        return lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)

    return _scan(f, init, xs, length, reverse, unroll)


def conv1d_lax(signal, kernel, mode='SAME'):
    return _conv1d_lax(signal, kernel, mode)


@partial(jit, static_argnums=(2,))
def _conv1d_lax(signal, kernel, mode):
    '''
    CPU impl. is insanely slow for large kernels, jaxlib-cuda (i.e. cudnn's GPU impl.)
    is highly recommended
    '''
    x = device_put(signal)
    h = device_put(kernel)

    x = x[jnp.newaxis,:,jnp.newaxis]
    h = h[::-1,jnp.newaxis,jnp.newaxis]
    dn = lax.conv_dimension_numbers(x.shape, h.shape, ('NWC', 'WIO', 'NWC'))

    # lax.conv_general_dilated runs much slower than numpy.convolve on CPU_device
    x = lax.conv_general_dilated(x,      # lhs = image tensor
                                 h,      # rhs = conv kernel tensor
                                 (1,),   # window strides
                                 mode.upper(), # padding mode
                                 (1,),   # lhs/image dilation
                                 (1,),   # rhs/kernel dilation
                                 dn)     # dimension_numbers = lhs, rhs, out dimension permu

    return x[0,:,0]


def _largest_prime_factor(n):
    '''brute-force finding of greatest prime factor of integer number.
    '''
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def _fft_size_factor(x, gpf):
    '''calculates the integer number exceeding parameter x and containing
    only the prime factors not exceeding gpf.
    '''
    if x <= 0:
        raise ValueError("The input value for factor is not positive.")
    x = int(x) + 1

    if gpf > 1:
        while(_largest_prime_factor(x) > gpf):
            x += 1

    return x


def conv1d_oa_fftsize(signal_length, kernel_length, oa_factor=10, max_fft_prime_factor=5):
    target_fft_size = kernel_length * oa_factor
    if target_fft_size < signal_length:
        fft_size = _fft_size_factor(target_fft_size, max_fft_prime_factor)
    else:
        fft_size = _fft_size_factor(max(signal_length, kernel_length), max_fft_prime_factor)

    return fft_size


def _conv1d_fft_oa_same(signal, kernel, fft_size):
    signal = device_put(signal)
    kernel = device_put(kernel)

    kernel_length = kernel.shape[-1] # kernel/filter length

    signal = _conv1d_fft_oa_full(signal, kernel, fft_size)

    signal = signal[(kernel_length - 1) // 2 : -(kernel_length // 2)]

    return signal


def _conv1d_fft_oa_full(signal, kernel, fft_size):
    ''' fast 1d convolute underpinned by FFT and overlap-and-add operations
    '''
    if isfloat(signal) and isfloat(kernel):
        fft = jnp.fft.rfft
        ifft = jnp.fft.irfft
    else:
        fft = jnp.fft.fft
        ifft = jnp.fft.ifft

    signal = device_put(signal)
    kernel = device_put(kernel)

    signal_length = signal.shape[-1]
    kernel_length = kernel.shape[-1]

    output_length = signal_length + kernel_length - 1
    frame_length = fft_size - kernel_length + 1

    frames = -(-signal_length // frame_length)

    signal = jnp.pad(signal, [0, frames * frame_length - signal_length])

    signal = jnp.reshape(signal, [-1, frame_length])
    signal = jnp.pad(signal, [0, fft_size - frame_length])
    kernel = jnp.pad(kernel, [0, fft_size - kernel_length])

    signal = ifft(fft(signal) * fft(kernel))
    signal = overlap_and_add(signal, frame_length)

    signal = signal[:output_length]

    return signal


@partial(jit, static_argnums=(2,3))
def _conv1d_fft_oa(signal, kernel, fft_size, mode):
    if mode.lower() == 'same':
        return _conv1d_fft_oa_same(signal, kernel, fft_size)
    else:
        return _conv1d_fft_oa_full(signal, kernel, fft_size)


def conv1d_fft_oa(signal, kernel, fft_size=None, oa_factor=10, mode='SAME'):
    if fft_size is None:
        signal_length = signal.shape[-1]
        kernel_length = kernel.shape[-1]
        fft_size = conv1d_oa_fftsize(signal_length, kernel_length, oa_factor=oa_factor)

    return _conv1d_fft_oa(signal, kernel, fft_size, mode)


def _frame_pad(array, flen, fstep, pad_constants):
    n = array.shape[0]
    fnum = -(-n // fstep) # double negatives to round up
    pad_len = (fnum - 1) * fstep + flen - n
    pad_width = ((0,pad_len),) + ((0,0),) * (array.ndim-1)
    array = jnp.pad(array, pad_width, constant_values=pad_constants)

    ind = jnp.arange(flen)[None,:] + fstep * jnp.arange(fnum)[:,None]
    return array[ind,...]


def _frame_direct(array, flen, fstep):
    n = array.shape[0]
    fnum = 1 + (n - flen) // fstep
    array = array[:(fnum - 1) * fstep + flen,...]

    ind = jnp.arange(flen)[None,:] + fstep * jnp.arange(fnum)[:,None]
    return array[ind,...]


@partial(jit, static_argnums=(1,2,3))
def _frame(array, flen, fstep, pad_end, pad_constants):
    n = array.shape[0]

    if n < flen:
        raise ValueError('array length {} < frame length {}'.format(n, flen))

    if flen < fstep:
        raise ValueError('frame length {} < frame step {}'.format(flen, fstep))

    if pad_end:
        return _frame_pad(array, flen, fstep, pad_constants)
    else:
        return _frame_direct(array, flen, fstep)


def frame(x, flen, fstep, pad_end=False, pad_constants=0.):
    return _frame(x, flen, fstep, pad_end, pad_constants)


def framescaninterp(x, func, flen, fstep, P=1):
    fn = lambda carry, y: (carry, func(y))
    N = x.shape[0]
    xf = frame(x, flen, fstep, pad_end=True)
    F = xf.shape[0]
    _, ys = scan(fn, None, xf)
    xp = jnp.arange(F) * fstep + flen // 2
    x = jnp.arange(N * P) / P
    interp = vmap(lambda x, xp, fp: jnp.interp(x, xp, fp), in_axes=(None, None, -1), out_axes=-1)
    ysip = interp(x, xp, ys) / P
    return ysip


@partial(jit, static_argnums=(1,))
def overlap_and_add(array, frame_step):
    array_shape = array.shape
    frame_length = array_shape[1]
    frames = array_shape[0]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # If frame_length is equal to frame_step, there's no overlap so just
    # reshape the tensor.
    if (frame_step == frame_length):
      return jnp.reshape(array, (output_length,))

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)
    paddings = [[0, segments], [0, segments * frame_step - frame_length]]
    array = jnp.pad(array, paddings)

    # Reshape
    array = jnp.reshape(array, [frames + segments, segments, frame_step])

    array = jnp.transpose(array, [1, 0, 2])

    shape = [(frames + segments) * segments, frame_step]
    array = jnp.reshape(array, shape)

    array = array[..., :(frames + segments - 1) * segments, :]

    shape = [segments, (frames + segments - 1), frame_step]
    array = jnp.reshape(array, shape)

    # Now, reduce over the columns, to achieve the desired sum.
    array = jnp.sum(array, axis=0)

    # Flatten the array.
    shape = [(frames + segments - 1) * frame_step]
    array = jnp.reshape(array, shape)

    # Truncate to final length.
    array = array[:output_length]

    return array


def delay(x, d):
    return jnp.roll(x, d, axis=0)


@jit
def finddelay(x, y):
    '''
    case 1:
        X = jnp.array([1, 2, 3])
        Y = jnp.array([0, 0, 1, 2, 3])
        D = xcomm.finddelay(X, Y) # D = 2
    case 2:
        X = jnp.array([0, 0, 1, 2, 3, 0])
        Y = jnp.array([0.02, 0.12, 1.08, 2.21, 2.95, -0.09])
        D = xcomm.finddelay(X, Y) # D = 0
    case 3:
        X = jnp.array([0, 0, 0, 1, 2, 3, 0, 0])
        Y = jnp.array([1, 2, 3, 0])
        D = xcomm.finddelay(X, Y) # D = -3
    case 4:
        X = jnp.array([0, 1, 2, 3])
        Y = jnp.array([1, 2, 3, 0, 0, 0, 0, 1, 2, 3, 0, 0])
        D = xcomm.finddelay(X, Y) # D = -1
    reference:
        https://www.mathworks.com/help/signal/ref/finddelay.html
    '''
    x = device_put(x)
    y = device_put(y)
    c = jnp.abs(correlate(x, y, mode='full', method='fft'))
    k = jnp.arange(-y.shape[0]+1, x.shape[0])
    i = jnp.lexsort((jnp.abs(k), -c))[0] # lexsort to handle case 4
    d = -k[i]
    return d, c


def fftconvolve(x, h, mode='full'):
    x = jnp.atleast_1d(x)
    h = jnp.atleast_1d(h)

    mode = mode.lower()

    if x.shape[0] < h.shape[0]:
        tmp = x
        x = h
        h = tmp

    T = h.shape[0]
    N = x.shape[0] + T - 1
    y = _fftconvolve(x, h)

    if mode == 'full':
        return y
    elif mode == 'same':
        return y[(T - 1) // 2:N - (T - 1) // 2]
    elif mode == 'valid':
        return y[T - 1:N - T + 1]
    else:
        raise ValueError('invalid mode')


@jit
def _fftconvolve(x, h):
    if isfloat(x) and isfloat(h):
        fft = jnp.fft.rfft
        ifft = jnp.fft.irfft
    else:
        fft = jnp.fft.fft
        ifft = jnp.fft.ifft

    N = x.shape[0]
    M = h.shape[0]

    out_length = N + M -1

    fft_size = _fft_size_factor(out_length, 5)

    x = jnp.pad(x, [0, fft_size - N])
    h = jnp.pad(h, [0, fft_size - M])

    X = fft(x)
    H = fft(h)

    y = ifft(X * H)

    y = y[:out_length]

    return y


def convolve(a, v, mode='full', method='auto'):
    a = jnp.atleast_1d(a)
    v = jnp.atleast_1d(v)
    method = method.lower()

    if method == 'auto':
        method = 0
    elif method == 'fft':
        method = 0
    elif method == 'direct':
        method = 1
    else:
        raise ValueError('invalid method')

    if method == 0:
        z = fftconvolve(a, v, mode=mode)
    else:
        # jnp.convolve() does not support complex number yet
        # jnp.convolve seems faster in 1d case, need to measure for precision
        if jnp.iscomplexobj(a) or jnp.iscomplexobj(v):
            z = conv1d_lax(a, v, mode)
        else:
            z = jnp.convolve(a, v, mode)
    return z


def correlate(a, v, mode='same', method='auto'):
    '''
    mode = 'same'
    c_{av}[k] = sum_n a[n+k] * conj(v[n])
    '''
    # NOTE: jnp.correlate() does not support complex inputs
    a = jnp.atleast_1d(a)
    v = jnp.atleast_1d(v)
    z = convolve(a, v[::-1].conj(), mode=mode, method=method)
    return z


def frft(f, a):
    """
    fast fractional fourier transform.
    Parameters
        f : [jax.]numpy array
            The signal to be transformed.
        a : float
            fractional power
    Returns
        data : [jax.]numpy array
            The transformed signal.
    reference:
        https://github.com/nanaln/python_frft
    """
    f = device_put(f)
    a = device_put(a)

    ret = jnp.zeros_like(f, dtype=jnp.complex64)
    f = f.astype(jnp.complex64)
    N = f.shape[0]

    shft = jnp.fmod(jnp.arange(N) + jnp.fix(N / 2), N).astype(int)
    sN = jnp.sqrt(N)
    a = jnp.remainder(a, 4.0)

    TRUE = jnp.array(True)
    FALSE = jnp.array(False)

    # simple cases
    ret, done = lax.cond(
        a == 0.0,
        None,
        lambda _: (f, TRUE),
        None,
        lambda _: (ret, FALSE))

    ret, done = lax.cond(
        a == 2.0,
        None,
        lambda _: (jnp.flipud(f), TRUE),
        None,
        lambda _: (ret, done))

    ret, done = lax.cond(
        a == 1.0,
        None,
        lambda _: (index_update(ret, index[shft], jnp.fft.fft(f[shft]) / sN), TRUE),
        None,
        lambda _: (ret, done))

    ret, done = lax.cond(
        a == 3.0,
        None,
        lambda _: (index_update(ret, index[shft], jnp.fft.ifft(f[shft]) * sN), TRUE),
        None,
        lambda _: (ret, done))

    @jit
    def sincinterp(x):
        N = x.shape[0]
        y = jnp.zeros(2 * N -1, dtype=x.dtype)
        y = index_update(y, index[:2 * N:2], x)
        xint = fftconvolve(
           y[:2 * N],
           jnp.sinc(jnp.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
        )
        return xint[2 * N - 3: -2 * N + 3]

    @jit
    def chirp_opts(a, f):
        # the general case for 0.5 < a < 1.5
        alpha = a * jnp.pi / 2
        tana2 = jnp.tan(alpha / 2)
        sina = jnp.sin(alpha)
        f = jnp.hstack((jnp.zeros(N - 1), sincinterp(f), jnp.zeros(N - 1))).T

        # chirp premultiplication
        chrp = jnp.exp(-1j * jnp.pi / N * tana2 / 4 *
                         jnp.arange(-2 * N + 2, 2 * N - 1).T ** 2)
        f = chrp * f

        # chirp convolution
        c = jnp.pi / N / sina / 4
        ret = fftconvolve(
            jnp.exp(1j * c * jnp.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
            f,
        )
        ret = ret[4 * N - 4:8 * N - 7] * jnp.sqrt(c / jnp.pi)

        # chirp post multiplication
        ret = chrp * ret

        # normalizing constant
        ret = jnp.exp(-1j * (1 - a) * jnp.pi / 4) * ret[N - 1:-N + 1:2]

        return ret

    def other_cases(a, f):

        a, f = lax.cond(
            a > 2.0,
            None,
            lambda _: (a - 2.0, jnp.flipud(f)),
            None,
            lambda _: (a, f))

        a, f = lax.cond(
            a > 1.5,
            None,
            lambda _: (a - 1.0, index_update(f, index[shft], jnp.fft.fft(f[shft]) / sN)),
            None,
            lambda _: (a, f))

        a, f = lax.cond(
            a < 0.5,
            None,
            lambda _: (a + 1.0, index_update(f, index[shft], jnp.fft.ifft(f[shft]) * sN)),
            None,
            lambda _: (a, f))

        return chirp_opts(a, f)

    ret = lax.cond(
        done,
        None,
        lambda _: ret,
        None,
        lambda _: other_cases(a, f))

    return ret



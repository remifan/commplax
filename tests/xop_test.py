from commplax import xop
import numpy as np
from jax import random, numpy as jnp


def conv_input_complex(n, m):
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(1)
    k1, k2 = random.split(key1)
    k3, k4 = random.split(key2)
    x = random.normal(k1, (n,)) + 1j * random.normal(k2, (n,))
    h = random.normal(k3, (m,)) + 1j * random.normal(k4, (m,))
    return x, h


def conv_input_float(n, m):
    key1 = random.PRNGKey(0)
    k1, k2 = random.split(key1)
    x = random.normal(k1, (n,))
    h = random.normal(k2, (m,))
    return x, h


def test_convolve():
    for n, m in zip([1, 5, 5, 5, 5, 6, 6, 6, 1000, 1000, 1001, 1001],
                    [1, 1, 2, 3, 4, 2, 3, 4, 7,    8,    7,    8]):

        for mode in ['same', 'valid', 'full']:
            x, h = conv_input_complex(n, m)
            a = np.convolve(x, h, mode=mode)
            b = xop.convolve(x, h, mode=mode)
            assert np.allclose(a, b, rtol=2e-05), "\nn={}, m={}, mode={}".format(n, m, mode)

        for mode in ['same', 'valid', 'full']:
            x, h = conv_input_float(n, m)
            a = np.convolve(x, h, mode=mode)
            b = xop.convolve(x, h, mode=mode)
            assert np.allclose(a, b, rtol=1e-05, atol=5e-06), "\nn={}, m={}, mode={}".format(n, m, mode)



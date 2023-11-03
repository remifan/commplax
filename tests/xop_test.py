import numpy as np
from absl.testing import absltest
import jax
from commplax import xop
from jax import random as jr, numpy as jnp, config
from jax._src import test_util as jtu

def conv_input(s, n, m, dtype):
    key1 = jr.key(s)
    k1, k2 = jr.split(key1)
    x = jr.normal(k1, (n,), dtype=dtype)
    h = jr.normal(k2, (m,), dtype=dtype)
    return x, h

class ConvTest(jtu.JaxTestCase):
    @jtu.sample_product(
        s=[0, 1],
        n=[1, 5, 100, 1000],
        m=[1, 2, 3, 10],
        mode=['same', 'valid', 'full'],
        dtype=jtu.dtypes.inexact,
    )
    def testConvolveFloat(self, s, n, m, mode, dtype):
        x, h = conv_input(s, n, m, dtype)
        a = np.convolve(x, h, mode=mode)
        b = xop.convolve(x, h, mode=mode)
        self.assertAllClose(a, b)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

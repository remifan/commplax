import numpy as np
import numpy.random as npr
from absl.testing import absltest
from jax import numpy as jnp, random as jr, jit
from jax._src import test_util as jtu

from commplax import dist_matcher as dm


class DMTest(jtu.JaxTestCase):
    @jtu.sample_product(
        s=[0, 1, 2],
        M=[4, 8, 16, 64],
        N=[2**10, 2**12, 2**15]
    )
    @jtu.run_on_devices('cpu')
    def testCCDM(self, s, M, N):
        npr.seed(s)
        p = npr.uniform(size=(M,))
        p /= p.sum()

        freqs, rate, encode, decode = dm.CCDM_helper(p, max_no_bits=N, num_state_bits=16)

        key = jr.key(s)
        bits_enc = jr.randint(key, (rate[0],), 0, 2)

        syms = encode(bits_enc)
        bits_dec = decode(syms)

        self.assertAllClose(bits_enc, bits_dec)
        #TODO check freqs

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
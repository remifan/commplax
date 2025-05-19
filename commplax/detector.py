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
import jax
import matplotlib.pyplot as plt
from functools import cache
from jax import numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(2))
def _unpack(d, B, N):
    return jax.lax.scan(lambda x, _: (x // B, x % B), d,
                        jnp.arange(N), unroll=True)[1].T[..., ::-1]

def viterbi(I, L, const, tblen=None):
    # I: input size
    # L: memory
    # tblen: traceback length
    S = I**L # states number
    T = 5 * L + 1 if tblen is None else tblen # trackback length
    si = jnp.arange(S) # state indices
    ii = jnp.arange(I) # input indices
    # ps[s] contains all the state (indices) yielding to state s with input (indices) pi[s].
    ps = jnp.tile(si.reshape((-1, I)), (I, 1)) # shape: (S, I)
    pi = jnp.repeat(jnp.repeat(ii[:, None], I, axis=1), S//I, axis=0) # shape: (S, I)
    # u[s] contains all the states (values) yielding to state s with input (indices) pi[s].
    u = jax.vmap(lambda s: const[_unpack(ps[s] + pi[s] * S, I, L+1)])(si) # shape: (S, I, L)

    # initial states
    pm_0 = jnp.zeros(S)
    tr_s_0 = jnp.zeros((T, S), dtype=int)
    tr_x_0 = jnp.zeros((T, S), dtype=int)

    def step(carry, inp):
        pm_k, tr_s, tr_x = carry
        y_k, w_k = inp

        v_t = jnp.inner(u, w_k)
        bm_t = jnp.abs(v_t - y_k)**2 
        pm_tmp = pm_k[ps] + bm_t
        idx = jnp.argmin(pm_tmp, axis=1)
        pm_k = pm_tmp[si, idx]

        tr_s = jnp.roll(tr_s, -1, axis=0).at[-1].set(ps[si, idx])
        tr_x = jnp.roll(tr_x, -1, axis=0).at[-1].set(pi[si, idx])

        # traceback
        x = jax.lax.scan(lambda s, t: (tr_s[t, s], tr_x[t, s]),
                         jnp.argmin(pm_k),
                         jnp.arange(T-1, -1, -1),
                         unroll=True)[1][-1] # complete unroll for the benefit of GPU

        carry = (pm_k, tr_s, tr_x)
        out = x

        return carry, out

    @jax.jit
    def scan(ys, h, pm_0=pm_0, tr_s=tr_s_0, tr_x=tr_x_0):
        # TODO: handle time varying h
        K = ys.shape[0]
        w = jnp.pad(h, [0, L+1-h.shape[0]])
        ws = jnp.repeat(w[None, :], K, axis=0)
        carry = (pm_0, tr_s_0, tr_x_0)
        carry, x_hat = jax.lax.scan(step, carry, (ys, ws))
        return carry, x_hat

    return scan


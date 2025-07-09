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
from jax import numpy as jnp, scipy as jsp
from functools import partial


@partial(jax.jit, static_argnums=(2))
def _unpack(d, B, N):
    return jax.lax.scan(lambda x, _: (x // B, x % B), d,
                        jnp.arange(N), unroll=True)[1].T[..., ::-1]


def mlse_viterbi(I, L, const, btd=None):
    # I: input size (e.g. 2 for binary)
    # L: memory
    # btd: backtrace depth
    S = I**L # states number
    T = 5 * L + 1 if btd is None else btd # trackback length
    si = jnp.arange(S) # state indices
    ii = jnp.arange(I) # input indices
    # ps[s] contains all the previous states having a branch with state s with input pi[s]
    ps = jnp.tile(si.reshape((-1, I)), (I, 1)) # shape: (S, I)
    pi = jnp.repeat(jnp.repeat(ii[:, None], I, axis=1), S//I, axis=0) # shape: (S, I)
    # u[s] contains all the symbols yielding to state s with input pi[s].
    u = jax.vmap(lambda s: const[_unpack(ps[s] + pi[s] * S, I, L+1)])(si) # shape: (S, I, L+1)

    # initial states
    pm_0 = jnp.zeros(S)
    tr_s_0 = jnp.zeros((T, S), dtype=int)
    tr_x_0 = jnp.zeros((T, S), dtype=int)
    state_0 = (pm_0, tr_s_0, tr_x_0)

    def step(state, y_k):
        pm_k, tr_s, tr_x, v_k = state

        bm_k = jnp.abs(v_k - y_k)**2  # branch metirc
        pm_tmp = pm_k[ps] + bm_k
        idx = jnp.argmin(pm_tmp, axis=1)
        pm_k = pm_tmp[si, idx]

        tr_s = jnp.roll(tr_s, -1, axis=0).at[-1].set(ps[si, idx])
        tr_x = jnp.roll(tr_x, -1, axis=0).at[-1].set(pi[si, idx])

        # backtrace; T-1 delay is introduced.
        x = jax.lax.scan(lambda s, t: (tr_s[t, s], tr_x[t, s]),
                         jnp.argmin(pm_k),
                         jnp.arange(T-1, -1, -1),
                         unroll=True)[1][-1] # complete unroll for the benefit of GPU

        state = pm_k, tr_s, tr_x, v_k
        out = const[x]

        return state, out

    @jax.jit
    def scan(state, ys, h, pm_0=pm_0, tr_s=tr_s_0, tr_x=tr_x_0):
        w = jnp.pad(h, [0, L+1-h.shape[0]])
        v = jnp.inner(u, w)
        carry = state + (v,)
        carry, x_hat = jax.lax.scan(step, carry, ys)
        state = carry[:3]
        return state, x_hat

    return scan, state_0


def map_bcjr(y, I, L, const, priori=None):
    # I: input size
    # L: memory
    # N: traceback length
    S = I**L # states number
    si = jnp.arange(S) # state indices
    ii = jnp.arange(I) # input indices
    priori = jnp.ones(I)/I if priori is None else priori 
    # ps[s] contains all the state (indices) yielding to state s with input (indices) pi[s].
    ps = jnp.tile(si.reshape((-1, I)), (I, 1)) # shape: (S, I)
    pi = jnp.repeat(jnp.repeat(ii[:, None], I, axis=1), S//I, axis=0) # shape: (S, I)
    # u[s] contains all the states (values) yielding to state s with input (indices) pi[s].
    u = jax.vmap(lambda s: const[_unpack(ps[s] + pi[s] * S, I, L+1)])(si) # shape: (S, I, L+1)
 
    lse = jsp.special.logsumexp
 
    noise_std = 0.001
    w = jnp.pad(h, [0, L+1-h.shape[0]])
    v = jnp.inner(u, w)
 
    err = jnp.abs(y[:, None, None] - v)**2
    log_priori = jnp.log(priori + 1e-30)[pi]
    gamma = -err / (2 * noise_std**2) - jnp.log(np.sqrt(2 * np.pi) * noise_std) + log_priori
    # Forward recursion
    alpha = jax.lax.scan(
        lambda a, g: (lse((a[ps] + g), axis=1), a),
        jnp.full(S, 0), gamma,
    )[1]
    # Backward recursion
    beta = jax.lax.scan(
        lambda b, g: (lse((b[:, None] + g).reshape((I, I * S//I)), axis=0), b),
        jnp.full(S, 0), gamma[::-1, ...],
    )[1][::-1, ...]

    lp = (P:=lse((alpha[:,ps] + gamma + beta[..., None]).reshape((-1, I, S)), axis=2)) \
        - lse(P, axis=1)[:, None] # log normalization
    xi = jnp.argmax(lp, axis=1)
    x_hat = const[xi]
    llr = lp[:, :, None] - lp[:, None, :]
        
    return x_hat, llr


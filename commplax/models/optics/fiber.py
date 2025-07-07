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


from functools import partial
from jax import numpy as jnp, lax
import numpy as np


def manakovSSF(
    Ein,
    f_s,
    L,
    a=.2E-3,
    D_λ=16E-6,
    λ_ref=1552.52e-9,
    n_2=2.6E-20,
    A_eff=80E-12,
    hz_max=1e3,
    hz_min=.0,
    Φ_max=0.05):

    E_t = Ein["E"]
    f_s = Ein["fs"]
    f_c = Ein["fc"]

    Nfft = E_t.shape[0]

    π = jnp.pi
    fft = partial(jnp.fft.fft, axis=0)
    ifft = partial(jnp.fft.ifft, axis=0)
    fftfreq = jnp.fft.fftfreq

    c = 299792458
    α = a / (10. / jnp.log(10.))
    γ = 2 * π * n_2 / λ_ref / A_eff
    β_2 = -(D_λ * λ_ref**2)/(2 * π * c)
    ω = 2 * π * f_s * fftfreq(Nfft)
    
    Epow = lambda E: (jnp.abs(E)**2).sum(axis=-1)

    # calculate bounded step size given signal power
    calc_hz = lambda E_t: jnp.clip(Φ_max / (γ * 8/9 * Epow(E_t).max()),
                                  hz_min,
                                  hz_max)

    # linear (D) and nonlinear (N) operators
    lin_op = lambda E_f, hz: E_f * \
        jnp.exp(-(α / 2) * (hz / 2) + 1j * (β_2 / 2) * (ω[:, None]**2) * (hz / 2))

    nli_op = lambda E_t, hz: E_t * \
        jnp.exp(1j * γ * 8/9 * Epow(E_t) * hz)[:, None]

    # symmetric ssfm: ...(D-N-D)-(D-N-D)-..., where signal starts and ends at
    # frequency domain for each step
    hz_new = calc_hz(E_t)
    E_f = fft(E_t)

    def inner_step(carry, _):
        z, hz_new, E_f = carry
        # update split step size
        hz = jnp.minimum(hz_new, z)

        # 1st linear step (frequency domain)
        E_f = lin_op(E_f, hz)

        # Nonlinear step (time domain)
        E_t = ifft(E_f)
        E_t = nli_op(E_t, hz)

        # new step size is supposed to be calculated at beginning of each iteration,
        # which needs extra fft/ifft. Calculating step size at middle step should be
        # safe since signal power is unlikely to vary much after single nonlinear step.
        hz_new = calc_hz(E_t)
        
        # 2nd linear step (frequency domain)
        E_f = fft(E_t)
        E_f = lin_op(E_f, hz)

        z = z - hz

        carry = z, hz_new, E_f
        return carry, hz

    def outer_step(carry):
        carry, _ = lax.scan(inner_step, carry, jnp.empty(100)) # "always scan when you can"
        return carry

    *_, E_f = lax.while_loop(lambda c: c[0] > 0, outer_step, (L, hz_new, E_f))

    E_t = ifft(E_f)

    Eout = {
        "fc": f_c,
        "fs": f_s,
        "T": Ein["T"],
        "E": E_t,
    }

    return Eout


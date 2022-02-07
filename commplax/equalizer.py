# Copyright 2021 The Commplax Authors.
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
import haiku as hk
from jax import jit, device_put, numpy as jnp
from commplax import xop, comm, xcomm, util, adaptive_filter as af
from commplax.module import hk as chk
from jax.experimental import host_callback as hcb


c2r = af.c2r
r2c = af.r2c
train_schedule = af.train_schedule


def tddbp(signal, sr, lp, dist, spans, taps, xi=0.6, D=16.5E-6, polmux=True, mode='SAME', backend=util.gpufirstbackend()):
    x = device_put(signal)
    return jit(_tddbp,
               static_argnums=(4, 5, 8, 9),
               backend=backend)(x, sr, lp, dist, spans, taps, xi, D, polmux, mode)


def _tddbp(x, sr, lp, dist, spans, taps, xi, D, polmux, mode):
    x = xcomm.normpower(x)
    _, param_D, param_N = xcomm.dbp_params(sr, dist/spans, spans, taps, fiber_dispersion=D, polmux=polmux)
    return xcomm.dbp_timedomain(x, param_D,
                                xi * 10.**(lp / 10 - 3) * param_N,
                                mode=mode)


def cdctaps(sr, CD, fc=193.4e12, odd_taps=True):
    pi = np.pi
    C = 299792458. # speed of light [m/s]
    lambda_ = C / fc
    # estimate minimal required number of taps for single step TDDBP
    mintaps = int(np.ceil(2 * pi * CD * lambda_**2 / (2 * pi * C) * sr**2))
    if odd_taps and mintaps % 2 == 0:
        mintaps += 1
    return mintaps


def cdcomp(signal, sr, CD, fc=193.4e12, taps=None, polmux=True, mode='SAME', backend=None):
    D = 16.5E-6
    dist = CD / D
    if taps is None:
        taps = cdctaps(sr, CD, fc)
    return tddbp(signal, sr, 0., dist, 1, taps, xi=0., D=D, polmux=polmux, mode=mode, backend=backend)


def cma(x, lr=1/2**14, sps=2, taps=32, const="16QAM", tap_fn=lambda *a:None):
    x = jnp.asarray(x)
    xf = af.frame(x, taps, sps)
    cma = chk.CMA(lr=lr, const=const)
    y = cma(xf)
    hcb.id_tap(tap_fn, cma.aux_out)
    return y


def mucma(x, sps=2, lr=1/2**14, const="16QAM", taps=32, delta=6, tap_fn=lambda *a:None):
    x = jnp.asarray(x)
    xf = af.frame(x, taps, sps)
    mucma = chk.MUCMA(lr=lr, delta=delta, const=const)
    y = mucma(xf)
    hcb.id_tap(tap_fn, mucma.aux_out)
    return y


def ekffoe(x, block_size=100, const="16QAM", apply_to=None, apply_to_sps=2, tap_fn=lambda *a:None):
    x = jnp.asarray(x)
    w0 = qamfoe(x[:40000])[1]
    foe = chk.FrameEKFCPR(w0=w0, const=const)
    xf = xop.frame(x, block_size, block_size)
    foe(xf)
    ws = foe.aux_out['ws'].mean(axis=-1)
    w = jnp.interp(jnp.arange(x.shape[0]),
                   jnp.arange(ws.shape[0]) * block_size + (block_size - 1) / 2, ws)
    psi = jnp.cumsum(w)
    y = x * jnp.exp(-1j * psi)[:, None]
    if apply_to is not None:
        x = apply_to
        w = jnp.interp(jnp.arange(x.shape[0]) / apply_to_sps, 
                       jnp.arange(w.shape[0]),
                       w) / apply_to_sps
        psi = jnp.cumsum(w)
        z = x * jnp.exp(-1j * psi)[:, None]

    hcb.id_tap(tap_fn, {"w": w})

    return y if apply_to is None else (y, z)


def efkcpr(x, const="16QAM", tap_fn=lambda *a:None):
    x = jnp.asarray(x)
    cpr = chk.CPANEEKFCPR(const=const)
    z = cpr(x)
    hcb.id_tap(tap_fn, cpr.aux_out)
    return z


def ddlms(x, truth=None, train=False, const="16QAM", sps=2, taps=32, lr_w=1/2**4, lr_f=1/2**7, lr_b=1/2**11, tap_fn=lambda *a:None):
    x = jnp.asarray(x)
    xf = af.frame(x, taps, sps)
    lms = chk.DDLMS(train=train, const=const, lr_w=lr_w, lr_f=lr_f, lr_b=lr_b)
    z = lms(xf, truth)
    hcb.id_tap(tap_fn, lms.aux_out)
    return z


def qamfoe(x, local=False, fitkind=None, lfoe_fs=16384, lfoe_st=5000, backend='cpu'):
    x = device_put(x)
    return jit(_qamfoe,
               static_argnums=(1, 2, 3, 4),
               backend=backend)(x, local, fitkind, lfoe_fs, lfoe_st)


def _qamfoe(x, local, fitkind, lfoe_fs, lfoe_st):
    dims = x.shape[-1]

    if local:
        fo = xcomm.localfoe(x, frame_size=lfoe_fs, frame_step=lfoe_st, sps=1, fitkind=fitkind).mean(axis=-1)
        x *= jnp.tile(jnp.exp(-1j * jnp.cumsum(fo))[:, None], (1, dims))
    else:
        fo = xcomm.foe_mpowfftmax(x)[0].mean()
        T = jnp.arange(x.shape[0])
        x *= jnp.tile(jnp.exp(-1j * fo * T)[:, None], (1, dims))

    return x, fo


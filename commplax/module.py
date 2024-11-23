import dataclasses as dc
import numpy as np
from jax import lax, numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any
import equinox as eqx
from equinox import field
from commplax import adaptive_filter_ as _af, xcomm
from commplax.util import default_complexing_dtype, default_floating_dtype, astuple


def scan(mod, xs, filter=eqx.is_array):
    arr, static = eqx.partition(mod, filter)
    def step(carry, x):
        mod = eqx.combine(carry, static)
        mod, y = mod(x)
        carry, _ = eqx.partition(mod, filter)
        return carry, y
    arr, ys = lax.scan(step, arr, xs)
    mod = eqx.combine(arr, static)
    return mod, ys


class MIMOCell(eqx.Module):
    fifo: Array
    state: Array
    cnt: int
    sps: int = field(static=True)
    af: PyTree = field(static=True)
    decimate: bool = field(static=True)

    def __init__(
        self,
        num_taps: int = 15,
        dims: int = 1,
        dtype=None,
        sps: int = 1,
        af: PyTree = None,
        state: Array = None,
        fifo: Array = None,
        cnt: int = 0,
        decimate: bool = False
    ):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.sps = sps
        self.af = _af.lms() if af is None else af
        self.state = self.af.init(taps=num_taps, dims=dims) if state is None else state
        self.fifo = jnp.zeros((num_taps, dims), dtype=dtype) if fifo is None else fifo
        self.cnt = jnp.asarray(cnt)
        self.decimate = decimate

    def __call__(self, input: PyTree):
        x, *args = astuple(input)
        shift = self.sps if self.decimate else 1
        fifo = jnp.roll(self.fifo, -shift, axis=0).at[-shift:].set(x)
        output = self.af.apply(self.state, fifo)
        state = lax.cond(
            self.decimate | (self.cnt % self.sps == 0),
            lambda *_: self.af.update(self.cnt, self.state, (fifo, *args))[0],
            lambda *_: self.state,
            )
        cell = dc.replace(self, fifo=fifo, state=state, cnt=self.cnt+1)
        return cell, output


class MIMO(eqx.Module):
    cell: eqx.Module

    def __init__(self, cell: eqx.Module):
        self.cell = cell

    def __call__(self, input):
        cell, output = scan(self.cell, input)
        mimo = dc.replace(self, cell=cell)
        return mimo, output


class FOE(eqx.Module):
    fo: float
    metric: Array
    cnt: int
    beta: float
    ts: Array

    def __init__(self, T=1000, fo=0., metric=None, cnt=0, beta=0.9, ts=None):
        self.ts = jnp.arange(T) if ts is None else ts
        self.cnt = cnt
        self.beta = beta
        self.metric = jnp.zeros(T, dtype=default_floating_dtype()) if metric is None else metric
        self.fo = fo

    def __call__(self, input):
        foe = self.detect(input)
        output = foe.equalize(input)
        return foe, output

    def detect(self, input):
        fo, metric = xcomm.foe_mpowfftmax(input)
        fo = fo.mean()
        metric = metric.mean(axis=-1)
        fo = self.beta * self.fo + (1-self.beta) * fo
        foe = dc.replace(self, fo=fo, metric=metric, cnt=self.cnt+1)
        return foe

    def equalize(self, input):
        T = self.cnt + self.ts
        output = input * jnp.exp(-1j * self.fo * T)[:, None]
        return output


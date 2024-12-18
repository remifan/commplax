import dataclasses as dc
import numpy as np
import jax
from jax import lax, numpy as jnp
import equinox as eqx
from equinox import field
from commplax import adaptive_filter as _af, xcomm
from commplax.util import default_complexing_dtype, default_floating_dtype, astuple
from functools import lru_cache, wraps, partial
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any
from typing import Callable, TypeVar
from numpy.typing import ArrayLike
import types
from inspect import isclass


dim_ax = eqx.if_array(-1)
M = TypeVar('M', bound=Callable)
T = TypeVar('T', bound=Callable)

def make_ensamble(
        cls: M, 
        reps: ArrayLike, 
        transform: T = eqx.filter_vmap(in_axes=dim_ax, out_axes=dim_ax),
        func=['__call__']) -> M:
    _spec = {f: transform(getattr(cls, f)) for f in func} # TODO: exception handling
    _cls = type(cls.__name__+'_ensamble', (cls,), _spec) # to avoid monkey patching
    @wraps(cls) # preserve the metadata and typing
    def wrapper(*args, **kwargs):
        return transform(lambda _: _cls(*args, **kwargs))(jnp.empty(reps))
    return wrapper


def make_iterable(mod: M, func=['__call__'], **scan_kwargs) -> M:
    if isclass(mod):
        mod = _make_iterable(mod, func=func, **scan_kwargs)
    else:
        cls = mod.__class__
        cls = _make_iterable(cls, func=func, **scan_kwargs)
        mod = cls(**dc.asdict(mod))
    return mod


def _make_iterable(cls: M, func=['__call__'], **scan_kwargs) -> M:
    def make_wrapper(f):
        @wraps(f) # preserve the metadata and typing
        def wrapper(self, xs):
            return scan(self, xs, func=f, **scan_kwargs)
        return wrapper
    isstr = lambda x: isinstance(x, str)
    cls_dict = {f: make_wrapper(getattr(cls, f) if isstr(f) else f) for f in func}
    _cls = type(cls.__name__+'_scan', (cls,), cls_dict) # to avoid monkey patching
    return _cls


def scan(mod, xs, filter=eqx.is_array, func=None, cb=None):
    func = getattr(mod.__class__, '__call__') if func is None else func
    arr, static = eqx.partition(mod, filter)
    def step(carry, x):
        mod = eqx.combine(carry, static)
        mod, y = func(mod, x)
        if cb is not None:
            jax.debug.callback(cb, mod)
        carry, _ = eqx.partition(mod, filter)
        return carry, y
    arr, ys = lax.scan(step, arr, xs)
    mod = eqx.combine(arr, static)
    return mod, ys


# MIMO = make_iterable(MIMOCell) # maybe buggy


class FOE(eqx.Module):
    fo: float
    i: int
    t: int
    state: PyTree
    af: PyTree = field(static=True)
    uar: float = field(static=True)
    mode: str = field(static=True)

    def __init__(self, fo=0.0, uar=1.0, af=None, i=0, t=0, mode="feedforward", state=None, af_kwds={}):
        self.i = jnp.asarray(i)
        self.t = jnp.asarray(t)
        self.af = _af.foe_YanW(**af_kwds) if af is None else af
        self.fo = jnp.asarray(fo)
        self.uar = uar * 1.0
        fo4init = fo if mode == "feedforward" else 0.
        self.state = self.af.init(fo4init) if state is None else state
        self.mode = mode

    def __call__(self, input):
        if self.mode == "feedforward":
            foe = self.update(input)[0]
            foe, output = foe.apply(input)
        else:
            foe, output = foe.apply(input)
            foe = foe.update(output)[0]
        return foe, output

    def update(self, input):
        state, out = self.af.update(self.i, self.state, input)
        fo = self.fo + out[0] if self.mode == "feedback" else out[0]
        foe = dc.replace(self, fo=fo, state=state, i=self.i+1)
        return foe, None

    def apply(self, input):
        T = self.t + jnp.arange(input.shape[0])
        fo = self.fo * self.uar
        output = input * jnp.exp(-1j * fo * T)
        foe = dc.replace(self, t=T[-1]+1)
        return foe, output


# class FOE_MIMO_LOOP(eqx.Module):
#     foe: FOE
#     mimo: MIMO
#     sps: int = field(static=True)

#     def __init__(self, sps=2, foe=None, mimo=None, dims=2, fo=0.):
#         self.foe = make_ensamble(FOE, dims, func=['apply', 'update'])(fo=fo, uar=1/sps, mode='feedback') if foe is None else foe
#         self.mimo = MIMO(19, af=_af.cma(lr=2**-13), sps=sps, dims=dims, decimate=False) if mimo is None else mimo
#         self.sps = sps

#     def __call__(self, x):
#         foe, y = self.foe.apply(x)
#         mimo, y = self.mimo(y)
#         foe = foe.update(y[::self.sps])[0]
#         foe = dc.replace(foe, fo=foe.fo.mean()*jnp.ones_like(foe.fo))
#         fml = dc.replace(self, foe=foe, mimo=mimo)
#         return fml, y

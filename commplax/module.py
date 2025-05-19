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


import dataclasses as dc
import numpy as np
import jax
import equinox as eqx
from jax import lax, numpy as jnp
from equinox import field
from functools import wraps
from jaxtyping import Array, Float, Int, PyTree
from typing import Callable, Any
from typing import Callable, TypeVar
from numpy.typing import ArrayLike
from inspect import isclass
from commplax.jax_util import default_complexing_dtype, default_floating_dtype, astuple


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
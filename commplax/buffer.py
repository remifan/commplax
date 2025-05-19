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
from typing import NamedTuple 
from jax import numpy as jnp, lax
from typing import Callable
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree
from commplax import jax_util as cu


class _SyncFIFO(NamedTuple):
    init: Callable
    apply: Callable


class _AsyncFIFO(NamedTuple):
    init: Callable
    enq: Callable
    deq: Callable
    size: Callable
    isfull: Callable
    isempty: Callable


class SyncFIFO(eqx.Module):
    op: _SyncFIFO = eqx.field(static=True)
    state: PyTree

    def __init__(self, shape=10, dtype=None, axis=0, state=None, op=None):
        dtype = cu.default_complexing_dtype() if dtype is None else dtype
        self.op = simple_sync_fifo(axis=axis) if op is None else op
        self.state = self.op.init(shape, dtype=dtype) if state is None else state

    def __call__(self, x):
        state, y = self.op.apply(self.state, x)
        return dc.replace(self, state=state), y


class AsyncFIFO(eqx.Module):
    op: _AsyncFIFO = eqx.field(static=True)
    state: PyTree
    watermark: Array

    def __init__(self, shape=10, dtype=None, watermark=0, axis=0, state=None, op=None):
        dtype = cu.default_complexing_dtype() if dtype is None else dtype
        self.op = simple_async_fifo(axis=axis) if op is None else op
        self.state = self.op.init(shape, dtype=dtype) if state is None else state
        self.watermark = jnp.asarray(watermark)

    def enq(self, x):
        state = self.op.enq(self.state, x)
        return dc.replace(self, state=state)

    def deq(self):
        state, y = self.op.deq(self.state)
        return dc.replace(self, state=state), y

    def size(self):
        return self.op.size(self.state)

    def isavail(self):
        return self.size() > self.watermark

    def isfull(self):
        return self.op.isfull(self.state)

    def isempty(self):
        return self.op.isempty(self.state)


def simple_sync_fifo(axis=0, shiftleft=True):
    def init(shape, init_val=0, dtype=cu.default_floating_dtype()):
        buffer = jnp.full(shape, init_val, dtype)
        state = buffer,
        return state

    def apply(state, x):
        buffer, = state
        x = jnp.expand_dims(x, axis) if x.ndim < buffer.ndim else x
        shift = x.shape[axis]
        buffer = jnp.moveaxis(buffer, axis, 0)
        if shiftleft:
            y = buffer[:shift]
            buffer = jnp.roll(buffer, -shift, axis=0).at[-shift:].set(x)
        else:
            y = buffer[-shift:]
            buffer = jnp.roll(buffer, shift, axis=0).at[:shift].set(x)
        buffer = jnp.moveaxis(buffer, 0, axis)
        state = buffer,
        return state, y

    return _SyncFIFO(init, apply)


def simple_async_fifo(axis=0):
    '''
    A circular FIFO implementation that favors in-place ops guaranteed by jax.jit [1].

    implications:
      its time order is messy if not being completely lost. Hence, this
        type of buffer should not be used in linear filters.

    behaviours:
      1. ignore any np.nan input 
      2. return np.nan on dequeuing empty buffer
      the second behaviour is used as "soft assertion" on empty queue since value-based
        checking is seen as errors by jax.jit.

    use cases:
      1. gearbox for multi-rate signal processing
      2. ...

    references:
      [1] https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    '''
    init_ptr = (jnp.array(-1), jnp.array(-1))

    def init(shape, init_val=0, dtype=cu.default_floating_dtype()):
        buffer = jnp.full(shape, init_val, dtype)
        pad_width = ((0,1),) + ((0,0),) * (buffer.ndim-1)
        buffer = jnp.pad(buffer, pad_width, constant_values=jnp.nan)
        front, rear = init_ptr
        state = buffer, front, rear
        return state

    def isfull(state):
        B, p, q = state
        n = B.shape[axis] - 1
        return (q + 1) % n == p

    def isempty(state):
        p = state[1]
        return p == -1

    def size(state):
        B, p, q = state
        n = B.shape[axis] - 1
        l = lax.select(p == -1, 0, (q - p) % n +1)
        return l

    def enqueue(state, x):
        x = jnp.asarray(x)
        state = lax.cond(jnp.isnan(x).any(), lambda *_: state, lambda *_: _enqueue(state, x))
        return state 

    def _enqueue(state, x):
        B, p, q = state
        n = B.shape[axis] - 1
        p = lax.select(p == -1, 0, p)
        q = (q + 1) % n
        B = B.at[q].set(x)
        state = B, p, q
        return state

    def dequeue(state):
        B, p, q = state
        n = B.shape[axis] - 1
        y = B[p]
        p, q = lax.cond(p == q, lambda *_: init_ptr, lambda *_: ((p + 1) % n, q))
        state = B, p, q
        return state, y

    return _AsyncFIFO(init, enqueue, dequeue, size, isfull, isempty)


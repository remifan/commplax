from typing import NamedTuple 
from jax import numpy as jnp, lax
from commplax import util as cu
from typing import Callable
from jaxtyping import Array, Float, Int, PyTree
import equinox as eqx


class _FIFO(NamedTuple):
    init: Callable
    enq: Callable
    deq: Callable
    size: Callable
    isfull: Callable
    isempty: Callable



class FIFO(eqx.Module):
    op: _FIFO
    state: PyTree
    watermark: Array

    def __init__(self, capacity=10, dims=2, dtype=None, watermark=0, state=None, op=None):
        dtype = default_complexing_dtype() if dtype is None else dtype
        self.op = circfifo() if op is None else op
        self.state = self.op.init((capacity, dims), dtype=dtype) if state is None else state
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


def circfifo(axis=0):
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

    def _isfull(state):
        B, p, q = state
        n = B.shape[axis] - 1
        return (q + 1) % n == p

    def _isempty(state):
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

    return _FIFO(init, enqueue, dequeue, size, _isfull, _isempty)


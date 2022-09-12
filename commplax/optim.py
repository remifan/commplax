# Copyright 2021 The Google and Commplax Authors.
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

# derived from jax.experimental.optimizer, Google owns its original copyright


from typing import Any, Callable, NamedTuple, Tuple, Union

from collections import namedtuple
import functools

import jax.numpy as jnp
from jax._src.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

map = safe_map
zip = safe_zip

# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

OptimizerState = namedtuple("OptimizerState",
                            ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]))


Array = Any
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
State = Any   # internal State
Updates = Params  # Gradient updates are of the same type as parameters.

InitFn = Callable[[Params], OptimizerState]
Step = int
UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
ParamsFn = Callable[[OptimizerState], Params]

class Optimizer(NamedTuple):
  init_fn: InitFn
  update_fn: UpdateFn
  params_fn: ParamsFn

Schedule = Callable[[Step], float]

# universal complex number handler that treats complex number as real-valued pair
# https://github.com/google/jax/issues/7357#issuecomment-885990271
def _iscomplex(x):
    return issubclass(x.dtype.type, jnp.complexfloating)
def _cx2flt(c):
    # convert a complex-valued tree to a pair of real-valued trees
    return tree_map(lambda *xs: tuple(xs), *tree_map(lambda x: (x.real, x.imag), c))
def _flt2cx(f):
    return tree_map(lambda a, b: a + 1j * b, *f)
def complex_as_realtuple(update_fn):
    def _upd(i, grad, state):
        if _iscomplex(grad):
            gr, gi = _cx2flt(grad)
            sr, si = _cx2flt(state)
            new_state = _flt2cx((update_fn(i, gr, sr), update_fn(i, -gi, si)))
        else:
            new_state = update_fn(i, grad, state)
        return new_state
    return _upd

def optimizer(opt_maker: Callable[...,
  Tuple[Callable[[Params], State],
        Callable[[Step, Updates, Params], Params],
        Callable[[State], Params]]],
  complex_handler=complex_as_realtuple) -> Callable[..., Optimizer]:
  """Decorator to make an optimizer defined for arrays generalize to containers.

  With this decorator, you can write init, update, and get_params functions that
  each operate only on single arrays, and convert them to corresponding
  functions that operate on pytrees of parameters. See the optimizers defined in
  optimizers.py for examples.

  Args:
    opt_maker: a function that returns an ``(init_fun, update_fun, get_params)``
      triple of functions that might only work with ndarrays, as per

      .. code-block:: haskell

          init_fun :: ndarray -> OptStatePytree ndarray
          update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
          get_params :: OptStatePytree ndarray -> ndarray

  Returns:
    An ``(init_fun, update_fun, get_params)`` triple of functions that work on
    arbitrary pytrees, as per

    .. code-block:: haskell

          init_fun :: ParameterPytree ndarray -> OptimizerState
          update_fun :: OptimizerState -> OptimizerState
          get_params :: OptimizerState -> ParameterPytree ndarray

    The OptimizerState pytree type used by the returned functions is isomorphic
    to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
    instead as e.g. a partially-flattened data structure for performance.
  """
  if complex_handler is None:
      complex_handler = lambda f: f

  @functools.wraps(opt_maker)
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params = opt_maker(*args, **kwargs)

    @functools.wraps(init)
    def tree_init(x0_tree):
      x0_flat, tree = tree_flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
      return OptimizerState(states_flat, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      states_flat, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(partial(complex_handler(update), i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      return OptimizerState(new_states_flat, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    return Optimizer(tree_init, tree_update, tree_get_params)
  return tree_opt_maker


@optimizer
def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    return x0
  def update(i, g, x):
    return x - step_size(i) * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)


@optimizer
def momentum(step_size: Schedule, mass: float):
  """Construct optimizer triple for SGD with momentum.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    mass: positive scalar representing the momentum coefficient.
  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * velocity
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params


@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * g * g + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


@optimizer
def adabound(lr=1e-3, final_lr=1e-3, b1=0.9, b2=0.999, gamma=1e-3, eps=1e-8):
  """Construct optimizer triple for AdaBound.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    gamma: convergence speed of the bound functions (default: 1e-3)
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  sche_lr = make_schedule(lr)
  base_lr = sche_lr(0)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    b1pow = b1 ** (i + 1)
    b2pow = b2 ** (i + 1)
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * g * g + b2 * v  # Second moment estimate.
    lri = sche_lr(i)
    lr = lri * (1 - b2pow) / (1 - b1pow) # bias correction
    # scale final_lr with lr_scheduler
    flr = final_lr * lri / base_lr
    lb = flr * (1 - 1 / (gamma * (i + 1) + 1)) # lower bound
    ub = flr * (1 + 1 / (gamma * (i + 1))) # upper bound
    lr_bd = jnp.clip(lr / (jnp.sqrt(v) + eps), lb, ub)
    x = x - lr_bd * m
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


@optimizer
def adabelief(step_size, b1=0.9, b2=0.999, eps=1e-8, rectify=True, sma_threshold=5.):
  """Construct optimizer triple for AdaBelief

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
    rectify: optional, boolean. Whether to enable rectification as in RectifiedAdam
    sma_threshold: optional. A float value. The threshold for simple mean average.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    s0 = jnp.zeros_like(x0)
    return x0, m0, s0
  def update(i, g, state):
    x, m, s = state
    b1pow = b1 ** (i + 1)
    b2pow = b2 ** (i + 1)
    sma_inf = 2. / (1. - b2) - 1.
    sma = sma_inf - 2. * i * b2pow / (1. - b2pow)
    m = (1. - b1) * g + b1 * m  # First  moment estimate.
    s = (1. - b2) * (g - m) * (g - m) + b2 * s + eps  # Second moment estimate.
    r = jnp.sqrt((sma - 4.) / (sma_inf - 4.) * (sma - 2.) / (sma_inf - 2.) * sma_inf / sma)
    mhat = m / (1. - b1pow)  # Bias correction.
    shat = s / (1. - b2pow)
    var = mhat / (jnp.sqrt(shat) + eps)
    if rectify:
        var = jnp.where(sma >= sma_threshold, r * var, mhat)
    x = x - step_size(i) * var
    return x, m, s
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


### learning rate schedules

def constant(step_size) -> Schedule:
  def schedule(i):
    return step_size
  return schedule

def exponential_decay(step_size, decay_steps, decay_rate):
  def schedule(i):
    return step_size * decay_rate ** (i / decay_steps)
  return schedule

def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
  if staircase:
    def schedule(i):
      return step_size / (1 + decay_rate * jnp.floor(i / decay_steps))
  else:
    def schedule(i):
      return step_size / (1 + decay_rate * i / decay_steps)
  return schedule

def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):
  def schedule(step_num):
    step_num = jnp.minimum(step_num, decay_steps)
    step_mult = (1 - step_num / decay_steps) ** power
    return step_mult * (step_size - final_step_size) + final_step_size

  return schedule

def piecewise_constant(boundaries: Any, values: Any):
  boundaries = jnp.array(boundaries)
  values = jnp.array(values)
  if not boundaries.ndim == values.ndim == 1:
    raise ValueError("boundaries and values must be sequences")
  if not boundaries.shape[0] == values.shape[0] - 1:
    raise ValueError("boundaries length must be one shorter than values length")

  def schedule(i):
    return values[jnp.sum(i > boundaries)]
  return schedule

def make_schedule(scalar_or_schedule: Union[float, Schedule]) -> Schedule:
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif jnp.ndim(scalar_or_schedule) == 0:
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))


### utilities

def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return tree_map(normalize, grad_tree)


### serialization utilities

class JoinPoint(object):
  """Marks the boundary between two joined (nested) pytrees."""
  def __init__(self, subtree):
    self.subtree = subtree

  # Since pytrees are containers of numpy arrays, look iterable.
  def __iter__(self):
    yield self.subtree

def unpack_optimizer_state(opt_state):
  """Converts an OptimizerState to a marked pytree.

  Converts an OptimizerState to a marked pytree with the leaves of the outer
  pytree represented as JoinPoints to avoid losing information. This function is
  intended to be useful when serializing optimizer states.

  Args:
    opt_state: An OptimizerState
  Returns:
    A pytree with JoinPoint leaves that contain a second level of pytrees.
  """
  states_flat, tree_def, subtree_defs = opt_state
  subtrees = map(tree_unflatten, subtree_defs, states_flat)
  sentinels = [JoinPoint(subtree) for subtree in subtrees]
  return tree_util.tree_unflatten(tree_def, sentinels)

def pack_optimizer_state(marked_pytree):
  """Converts a marked pytree to an OptimizerState.

  The inverse of unpack_optimizer_state. Converts a marked pytree with the
  leaves of the outer pytree represented as JoinPoints back into an
  OptimizerState. This function is intended to be useful when deserializing
  optimizer states.

  Args:
    marked_pytree: A pytree containing JoinPoint leaves that hold more pytrees.
  Returns:
    An equivalent OptimizerState to the input argument.
  """
  sentinels, tree_def = tree_flatten(marked_pytree)
  assert all(isinstance(s, JoinPoint) for s in sentinels)
  subtrees = [s.subtree for s in sentinels]
  states_flat, subtree_defs = unzip2(map(tree_flatten, subtrees))
  return OptimizerState(states_flat, tree_def, subtree_defs)

# Copyright 2026 The Commplax Authors.
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


import re
import os
import jax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_structure, treedef_is_leaf
from jax.interpreters import xla
from jax import numpy as jnp
from commplax.third_party import namedtuple_pprint
from functools import partial, update_wrapper
from collections.abc import Iterable


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def default_complexing_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.complex128
    else:
        return jnp.complex64


def astuple(x):
    return tuple(x) if (isinstance(x, tuple) or isinstance(x, list)) else (x,)


def getdev(x):
    return x.device_buffer.device()


def devputlike(x, y):
    '''put x into the same device with y'''
    return jax.device_put(x, getdev(y))


def gpuexists():
    try:
        gpus = jax.devices('gpu')
    except RuntimeError:
        return False
    return len(gpus) != 0


def gpufirstbackend():
    '''
    NOTE: `backend` api is experimental feature,
    https://jax.readthedocs.io/en/latest/jax.html#jax.jit
    '''
    return 'gpu' if gpuexists() else 'cpu'


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, ys


def chain(fs, init, length=None):
    if callable(fs):
        fs = [fs] * length
    ret = init
    for f in fs:
        ret = f(ret)
    return ret


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def none_is_leaf(n): return treedef_is_leaf(tree_structure(n))


def tree_shape(x):
    return tree_map(lambda x: x.shape, x)


def tree_full(tree, value, is_leaf=none_is_leaf):
    return tree_map(lambda _: value, tree, is_leaf=is_leaf)


def tree_all(x):
    return all(jax.tree_flatten(x)[0])


def tree_any(x):
    return any(jax.tree_flatten(x)[0])


def tree_transpose(list_of_trees):
  """Convert a list of trees of identical structure into a single tree of lists."""
  return jax.tree.map(lambda *xs: list(xs), *list_of_trees)


tree_map_nl = partial(tree_map, is_leaf=none_is_leaf)


def tree_update_ignorenoneleaves(x, y):
    '''
    replace x's leaves with y's non-None leaves
    '''
    x_flat, x_tree = tree_flatten(x)
    # WORKAROUND to make tree_flatten recognize None-valued pytree nodes as pytree leaves,
    # so that we can update (1, 2, (3, 4)) by (None, None, (5, None)) without errors
    # see https://fossies.org/linux/tensorflow/tensorflow/compiler/xla/python/pytree.cc
    # note returned y_tree also see None type as * type now
    # not sure if filter `is_leaf=lambda n: not isinstance(n, tuple)` has side effects, be cautious!
    y_flat, y_tree = tree_flatten(y, is_leaf=lambda n: not isinstance(n, tuple))

    if x_tree != y_tree:
      msg = ("tree update function produced an output structure that "
             "did not match its input structure: input {} and output {}.")
      raise TypeError(msg.format(x_tree, y_tree))
    z_flat = map(lambda a, b: a if b is None else b, x_flat, y_flat)
    z_tree = tree_unflatten(x_tree, z_flat)
    return z_tree


def _tree_replace(tree, subtree, value, none_leaf=True):
    ''' work with namedtuple-like node '''
    subtree_def = tree_structure(subtree)
    is_subtree_def = lambda x: tree_structure(x) == subtree_def
    return tree_map(lambda x: tree_map(lambda _: value,
                                       x,
                                       is_leaf=none_is_leaf if none_leaf else None) if is_subtree_def(x) else x,
                    tree,
                    is_leaf=is_subtree_def)


def tree_replace(tree, subtrees, value, none_leaf=True):
    # subtrees must be list
    if not isinstance(subtrees, list):
        subtrees = [subtrees]
    return scan(lambda t, s: (_tree_replace(t, s, value, none_leaf=none_leaf), None), tree, subtrees)[0]


pprint = namedtuple_pprint.PrettyPrinter(indent=2).pprint


def clear_xla_cache():
    ''' 
    compile cache grows without bound, clear on finish otherwise Colab might complain 
    about insufficient RAM. TODO: try to reuse model during initialization to save compilation
    '''
    xla._xla_callable.cache_clear() 


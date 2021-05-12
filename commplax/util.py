import jax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from collections import namedtuple as _namedtuple


def getdev(x):
    return x.device_buffer.device()


def devputlike(x, y):
    '''put x into the same device with y'''
    return jax.device_put(x, getdev(y))


def gpuexists():
    gpus = jax.devices('gpu')
    return len(gpus) != 0


def gpufirstbackend():
    '''
    NOTE: `backend` api is experimental feature,
    https://jax.readthedocs.io/en/latest/jax.html#jax.jit
    '''
    return 'gpu' if gpuexists() else 'cpu'


def tree_shape(x):
    return tree_map(lambda x: x.shape, x)


def tree_like(x, value=None):
    x_flat, x_tree = tree_flatten(x)
    v_flat = (value,) * len(x_flat)
    return tree_unflatten(x_tree, v_flat)


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


def tree_homoreplace(tree, subtree, value):
    subtree_def = tree_flatten(subtree)[1]
    is_subtree = lambda x: tree_flatten(x)[1] == subtree_def
    return tree_map(lambda x: tree_map(lambda _: value, x) if is_subtree(x) else x,
                    tree,
                    is_leaf=is_subtree)


def namedtuple(name, keys, *args, **kwargs):
    ''' patch collections.namedtuple with extra pytree utilites '''
    return type(name, (_namedtuple(name, keys, *args, **kwargs),), {'apply': tree_homoreplace})


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, ys



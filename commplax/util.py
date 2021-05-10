import jax
from jax._src.util import partial, safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

map = safe_map
zip = safe_zip


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
    return tree_util.tree_map(lambda x: x.shape, x)


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


tree_update = tree_update_ignorenoneleaves


def isnamedtupleinstance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i)==str for i in fields)


def unpack_namedtuple(obj):
    if isinstance(obj, dict):
        return {key: unpack_namedtuple(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack_namedtuple(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: unpack_namedtuple(value) for key, value in obj._asdict().items() if not (isinstance(value, tuple) and len(value) == 0)}
    elif isinstance(obj, tuple):
        return tuple(unpack_namedtuple(value) for value in obj)
    else:
        return obj


def dict_flatten(d, parent_key='', sep='_'):
    try:
        from collections.abc import MutableMapping
    except ModuleNotFoundError:
        from collections import MutableMapping
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



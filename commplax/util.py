import jax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_structure
from commplax.third_party import namedtuple_pprint, flax_serialization


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


def tree_shape(x):
    return tree_map(lambda x: x.shape, x)


def tree_full(weights, value=True):
    return tree_map(lambda _: value, weights)


def tree_all(x):
    return all(jax.tree_flatten(x)[0])


def tree_any(x):
    return any(jax.tree_flatten(x)[0])


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


def _tree_homoreplace(tree, subtree, value):
    ''' work with namedtuple-like node '''
    subtree_def = tree_structure(subtree)
    is_subtree_def = lambda x: tree_structure(x) == subtree_def
    return tree_map(lambda x: tree_map(lambda _: value, x) if is_subtree_def(x) else x,
                    tree,
                    is_leaf=is_subtree_def)


def tree_homoreplace(tree, subtrees, value):
    # subtrees must be list
    if not isinstance(subtrees, list):
        subtrees = [subtrees]
    return scan(lambda t, s: (_tree_homoreplace(t, s, value), None), tree, subtrees)[0]


pprint = namedtuple_pprint.PrettyPrinter(indent=2).pprint


def passkwargs(kwargs_dict, **default_kwargs):
    assert isinstance(kwargs_dict, dict)
    kwargs_dict = dict(kwargs_dict)
    for k, v in default_kwargs.items():
        kwargs_dict.update({k: kwargs_dict.pop(k, v)})
    return kwargs_dict


# shortcuts of flax's serilization API
tree_serialize = flax_serialization.msgpack_serialize
tree_restore = flax_serialization.msgpack_restore
from_bytes = flax_serialization.from_bytes
to_bytes = flax_serialization.to_bytes


def dump(obj, filename):
    with open(filename, 'wb') as outfile:
        outfile.write(to_bytes(obj))


def load(target, filename):
    with open(filename, 'rb') as datfile:
        obj = from_bytes(target, datfile.read())
    return obj

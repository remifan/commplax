from collections import namedtuple
from jax import tree_util


def apply_trainable_grad(g_tree, b_tree):
    g_tree_def, b_tree_def = tree_util.tree_structure(g_tree), tree_util.tree_structure(b_tree)
    assert g_tree_def == b_tree_def, "gradients and trainable have mismatched tree structure"
    return tree_util.tree_map(lambda g_leaf, b_leaf: g_leaf if b_leaf else g_leaf * 0, g_tree, b_tree)


def trainable(weights, invert=False):
    return tree_util.tree_map(lambda _: not invert, weights)


# noun
_all = lambda t: t

# verb
_train = lambda n: lambda t: t.apply(n(t), True)
_freeze = lambda n: lambda t: t.apply(n(t), False)
_train.T = _freeze
_freeze.T = _train

# decorator
_only = lambda v: lambda n: lambda t: v(n)(v.T(_all)(t))

TrainableMap = namedtuple('TrainableMap', 'all train freeze only')
tmap = TrainableMap(all=_all,
                    train=_train,
                    freeze=_freeze,
                    only=_only)


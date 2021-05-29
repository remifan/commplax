from flax.core import Scope, init, apply
from jax import jit, random
from functools import reduce
from commplax.util import wrapped_partial as partial
from typing import NamedTuple, Optional, Any
from commplax.module import core
import operator


class Layer(NamedTuple):
    name: Optional[str]
    init: Any
    apply: Any
    core: Any
    mutable: Any


def make_layer(f, mutable=()):
    def _layer(layer_transform=lambda f: f, **kwargs):
        name = kwargs.pop('name', None)
        core_fun = layer_transform(partial(f, **kwargs))

        def init_fun(rng, *args, **kwargs):
            return init(core_fun)(rng, *args, **kwargs)

        def apply_fun(params, *args, **kwargs):
            return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

        return Layer(name, init_fun, apply_fun, core_fun, mutable)
    return _layer


def vmap(layer, **vmap_kwargs):
    return partial(layer, layer_transform=partial(core.vmap, **vmap_kwargs))


Conv1d = make_layer(core.conv1d)
MIMOConv1d = make_layer(core.mimoconv1d)
MIMOAF = make_layer(core.mimoaf, mutable=('af_state',))
FDBP = make_layer(core.fdbp)
SimpleFn = make_layer(core.simplefn)


def Serial(*layers, name='serial'):
    names, _, _, core_funs, mutables = zip(*layers)
    core_fun = core.serial(*zip(names, core_funs))
    mutable = reduce(operator.add, list(mutables))

    def init_fun(rng, *args, **kwargs):
        return init(core_fun)(rng, *args, **kwargs)

    def apply_fun(params, *args, **kwargs):
        return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

    return Layer(name, init_fun, apply_fun, core_fun, mutable)



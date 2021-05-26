import jax
from flax.core import Scope, init, apply
from functools import partial, reduce
from jax import jit, random
from typing import NamedTuple, Any
from commplax.module import core
import operator


class Layer(NamedTuple):
    init: Any
    apply: Any
    core: Any
    mutable: Any


def make_layer(f, mutable=()):
    def _layer(**kwargs):
        core_fun = partial(f, **kwargs)

        def init_fun(rng, *args, **kwargs):
            return init(core_fun)(rng, *args, **kwargs)

        def apply_fun(params, *args, **kwargs):
            return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

        return Layer(init_fun, apply_fun, core_fun, mutable)
    return _layer


Conv1d = make_layer(core.conv1d)
MIMOConv1d = make_layer(core.mimoconv1d)
MIMOAF = make_layer(core.mimoaf, mutable=('af_state',))
FDBP = make_layer(core.fdbp)


def Serial(*layers):
    _, _, core_funs, mutables = zip(*layers)
    core_fun = core.serial(*core_funs)
    mutable = reduce(operator.add, list(mutables))

    def init_fun(rng, *args, **kwargs):
        return init(core_fun)(rng, *args, **kwargs)

    def apply_fun(params, *args, **kwargs):
        return apply(core_fun, mutable=mutable)(params, *args, **kwargs)

    return Layer(init_fun, apply_fun, core_fun, mutable)



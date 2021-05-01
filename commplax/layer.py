import numpy as np
from functools import partial, wraps
from jax import numpy as jnp, jit
from typing import NamedTuple, Callable, Tuple, Any
from collections import namedtuple
from commplax import comm, xcomm, xop, adaptive_filter as af


class Layer(NamedTuple):
    name: str
    init: Callable
    apply: Callable
    trange: Tuple


class LayerInitAns(NamedTuple):
    outshape: tuple
    weights: Any
    state: Any


class LayerApplyAns(NamedTuple):
    out: Any
    state: Any


trange0 = (0, 0)


def layer(layer_maker, pure_fn=False, has_state=False):
    @wraps(layer_maker)
    def _layer_maker(*args, **kwargs):
        name, init, apply, trange = (layer_maker(*args, **kwargs) + (trange0,))[:4]

        trange = tuple([t for t in trange])

        @wraps(init)
        def _init(*args, **kwargs):
            ret = init(*args, **kwargs)
            if pure_fn:
                ret = (ret, (), ())
            elif not has_state:
                ret = ret + ((),)
            else:
                pass
            _assert_tuple_len(ret, 3)
            return LayerInitAns(*ret)

        @wraps(apply)
        def _apply(weights, inputs, states, trange=trange0, **kwargs):
            _assert_tuple_len(trange, 2)
            if pure_fn:
                outputs = jit(apply, static_argnums=1)(inputs, trange, **kwargs)
            elif not has_state:
                outputs = jit(apply, static_argnums=2)(weights, inputs, trange, **kwargs)
            else:
                outputs, states = jit(apply, static_argnums=3)(weights, inputs, states, trange, **kwargs)
            return LayerApplyAns(outputs, states)

        return Layer(name, _init, _apply, trange)

    return _layer_maker


def _assert_tuple_len(t, l):
    assert isinstance(t, tuple) and len(t) == l


def _slice_valid(value, trange):
    a = trange[0]
    b = trange[1]
    if value is not None: assert value.shape[0] > a - b
    value = value if value is None or a == 0 and b == 0 else value[a: b]
    return value


def _rename_dupnames(names):
    unames = list(set(names))
    cnt = dict(zip(unames, (0,) * len(unames)))
    renames = []
    for n in names:
        c = cnt[n]
        renames.append(n if c == 0 else n + str(c))
        cnt[n] += 1
    return renames


def _stp(names):
    ''' state namedtuple '''
    return namedtuple('State', names)


def _wtp(names):
    ''' weights namedtuple '''
    return namedtuple('Weights', names)


@partial(layer, pure_fn=True)
def FOE(sps=2, name='foe'):
    def init(input_shape):
        return input_shape

    def apply(inputs, trange, *args, fo=None, **kwargs):
        trange = (trange[0] * sps, trange[1] * sps)
        fo = _slice_valid(fo, trange)
        return inputs * fo

    return name, init, apply


@layer
def DBP(sr, lspan, nspan, dtaps, lp, sps=2, vspan=None, name='dbp'):
    steps = nspan if vspan is None else vspan
    n_invalid = (dtaps - 1) * steps
    trange = jnp.array([n_invalid // 2, -n_invalid // 2]) // sps

    def init(input_shape):
        _, wD, wN = comm.dbp_params(sr, lspan, nspan, dtaps, launch_power=lp, virtual_spans=vspan)
        output_shape = (input_shape[0] - n_invalid,) + input_shape[1:]
        weights = (wD, wN * 0.2)
        return output_shape, weights

    def apply(weights, inputs, *args, **kwargs):
        wD, wN = weights
        outputs = xcomm.dbp_timedomain(inputs, wD, wN, mode='valid')
        return outputs

    return name, init, apply, trange


@partial(layer, has_state=True)
def MIMO(taps=31, sps=2, train=False, mimo=af.ddlms, mimokwargs={}, mimoinitargs={}, name='mimo'):
    mimo_init, mimo_update, mimo_apply = mimo(train=train, **mimokwargs)
    trange = af.mimozerodelaypads(taps, sps)[0] // sps * np.array([1, -1])

    def init(input_shape):
        dims = input_shape[-1]
        stats = mimo_init(dims=dims, taps=taps, **mimoinitargs)
        output_shape = (xop.frame_shape(input_shape, taps, sps, allowwaste=True)[0], dims)
        states = (0, stats)
        return output_shape, (), states

    def apply(weights, inputs, states, trange, *args, **kwargs):
        truth = _slice_valid(kwargs.get('truth'), trange)
        inputs = xop.frame(inputs, taps, sps)
        i, stats = states
        i, (stats, (params, _)) = af.iterate(mimo_update, i, stats, inputs, truth)
        outputs = mimo_apply(params, inputs)
        states = (i + inputs.shape[0],) + stats
        return outputs, states

    return name, init, apply, trange


@partial(layer, pure_fn=True)
def MSE(name='mse'):
    def init(input_shape):
        return ()

    def apply(inputs, trange0, *args, truth=None, **kwargs):
        truth = _slice_valid(truth, trange0)
        return jnp.mean(jnp.abs(inputs - truth)**2)

    return name, init, apply


# Composing layers via combinators
@partial(layer, has_state=True)
def Serial(*layers, name='serial'):
    """Combinator for composing layers in serial.

    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
      composition of the given sequence of layers.
    """
    names, inits, applys, tranges = zip(*layers)
    nlayers = len(layers)
    names = _rename_dupnames(names)
    tranges = np.array(tranges)
    trange = tranges.sum(axis=0)
    cumtranges = tranges.cumsum(axis=0)

    def init(input_shape):
        weights = []
        states = []
        for init in inits:
            input_shape, weight, state = init(input_shape)
            weights.append(weight)
            states.append(state)
        return input_shape, _wtp(names)(*weights), _stp(names)(*states)

    def apply(weights, inputs, states, trange, **kwargs):
        # accumulate tranges
        tranges = [(tr[0], tr[1]) for tr in cumtranges + np.array(trange)[None, ...]]
        new_states = []
        for fun, weight, state, trange in zip(applys, weights, states, tranges):
            inputs, state = fun(weight, inputs, state, trange, **kwargs)
            new_states.append(state)
        return inputs, _stp(names)(*new_states)

    return name, init, apply, trange



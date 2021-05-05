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

        @partial(jit, static_argnums=3)
        @wraps(apply)
        def _apply_(weights, inputs, states, trange, **kwargs):
            if pure_fn:
                outputs = apply(inputs, trange, **kwargs)
            elif not has_state:
                outputs = apply(weights, inputs, trange, **kwargs)
            else:
                outputs, states = apply(weights, inputs, states, trange, **kwargs)
            return LayerApplyAns(outputs, states)

        def _apply(weights, inputs, states, trange=trange0, **kwargs):
            return _apply_(weights, inputs, states, trange, **kwargs)

        return Layer(name, _init, _apply, trange)

    return _layer_maker


fnlayer = partial(layer, pure_fn=True)
statlayer = partial(layer, has_state=True)


def _assert_tuple_len(t, l):
    assert isinstance(t, tuple) and len(t) == l


def _slice_valid(value, trange):
    a = trange[0]
    b = trange[1]
    if value is not None: assert value.shape[0] > a - b
    value = value if value is None or a == 0 and b == 0 else value[a: b]
    return value


def _addtrange(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _rename_dupnames(names):
    unames = list(set(names)) # dedup
    cnt = dict(zip(unames, (0,) * len(unames)))
    renames = []
    for n in names:
        c = cnt[n]
        renames.append(n if c == 0 else n + str(c))
        cnt[n] += 1
    return renames


def _itp(names):
    ''' input_shapes namedtuple '''
    return namedtuple('InputShape', names)


def _stp(names):
    ''' state namedtuple '''
    return namedtuple('State', names)


def _wtp(names):
    ''' weights namedtuple '''
    return namedtuple('Weights', names)


@fnlayer
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


@statlayer
def MIMO(taps=32, sps=2, train=False, mimo=af.ddlms, mimokwargs={}, mimoinitargs={}, name='mimo'):
    mimo_init, mimo_update, mimo_apply = mimo(train=train, **mimokwargs)
    trange = af.mimozerodelaypads(taps, sps)[0] // sps * np.array([1, -1])

    def init(input_shape):
        dims = input_shape[-1]
        stats = mimo_init(dims=dims, taps=taps, **mimoinitargs)
        output_shape = (xop.frame_shape(input_shape, taps, sps, allowwaste=True)[0], dims)
        states = (0, stats)
        return output_shape, (), states

    def apply(weights, inputs, states, abstrange, *args, **kwargs):
        truth = _slice_valid(kwargs.get('truth'), _addtrange(abstrange, trange))
        inputs = xop.frame(inputs, taps, sps)
        i, stats = states
        i, (stats, (params, _)) = af.iterate(mimo_update, i, stats, inputs, truth)
        outputs = mimo_apply(params, inputs)
        states = (i + inputs.shape[0],) + (stats,)
        return outputs, states

    return name, init, apply, trange


@fnlayer
def Downsample(sps=2, name='downsample'):
    def init(input_shape):
        return (input_shape[0] // sps,) + input_shape[1:]

    def apply(inputs, trange0, *args, **kwargs):
        return inputs[::sps]

    return name, init, apply


@fnlayer
def MSE(name='mse'):
    def init(input_shape):
        return ()

    def apply(inputs, trange0, *args, truth=None, **kwargs):
        truth = _slice_valid(truth, trange0)
        return jnp.mean(jnp.abs(inputs - truth)**2)

    return name, init, apply


@fnlayer
def Elementwise(fun, name='elementwise', **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init = lambda input_shape: input_shape
    apply = lambda inputs, *args, **kwargs: fun(inputs, **fun_kwargs)
    return name, init, apply


@fnlayer
def FanInConcat(axis=-1, name='fan_in_concat'):
    """Layer construction function for a fan-in concatenation layer."""
    def init(input_shape):
        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
        return out_shape
    def apply(inputs, *args, **kwargs):
        return jnp.concatenate(inputs, axis)
    return name, init, apply


@fnlayer
def FanInElementwise(fun, name='faninel', **fun_kwargs):
    init = lambda input_shape: input_shape[0]
    apply = lambda inputs, *args, **kwargs: fun(*inputs, **fun_kwargs)
    return name, init, apply


@fnlayer
def FanOut(num, name='fanout'):
  """Layer construction function for a fan-out layer."""
  init = lambda input_shape: [input_shape] * num
  apply = lambda inputs, *args, **kwargs: [inputs] * num
  return name, init, apply


# Composing layers via combinators
@statlayer
def Serial(*layers, name='serial'):
    """Combinator for composing layers in serial."""
    names, inits, applys, tranges = zip(*layers)
    nlayers = len(layers)
    names = _rename_dupnames(names)
    cumtranges = np.concatenate([np.array([trange0]), np.array(tranges).cumsum(axis=0)])
    trange, cumtranges = cumtranges[-1], cumtranges[:-1]

    # group sublayers into namedtuple
    WeightsTuple = _wtp(names)
    StateTuple = _stp(names)

    def init(input_shape):
        weights = []
        states = []
        for init in inits:
            input_shape, weight, state = init(input_shape)
            weights.append(weight)
            states.append(state)
        return input_shape, WeightsTuple(*weights), StateTuple(*states)

    def apply(weights, inputs, states, trange, **kwargs):
        # accumulate tranges
        tranges = [(tr[0], tr[1]) for tr in cumtranges + np.array(trange)[None, ...]]
        new_states = []
        for fun, weight, state, trange in zip(applys, weights, states, tranges):
            inputs, state = fun(weight, inputs, state, trange, **kwargs)
            new_states.append(state)
        return inputs, StateTuple(*new_states)

    return name, init, apply, trange


@statlayer
def Parallel(*layers, name='parallel'):
    """Combinator for composing layers in parallel."""
    nlayers = len(layers)
    names, inits, applys, tranges = zip(*layers)
    names = _rename_dupnames(names)

    # group sublayers into namedtuple
    InputsTuple = _itp(names)
    WeightsTuple = _wtp(names)
    StateTuple = _stp(names)

    def init(input_shape):
        input_shapes, weights, states = tuple(zip(*[init(shape) for init, shape in zip(inits, input_shape)]))
        return InputsTuple(*input_shapes), WeightsTuple(*weights), StateTuple(*states)

    def apply(weights, inputs, states, intranges, **kwargs):
        if isinstance(intranges, tuple) and isinstance(intranges[0], int):
            intranges = (intranges,)
        assert len(intranges) == nlayers
        intranges = [trange0 if itr == () else itr for itr in intranges]
        outputs, states = [f(w, x, s, (tr[0]+itr[0], tr[1] + itr[1]), **kwargs) for f, w, x, s, tr, itr in \
                           zip(applys, weights, inputs, states, tranges, intranges)]
        return outputs, StateTuple(*states)

    return name, init, apply, tranges



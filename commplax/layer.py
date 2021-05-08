import operator
import numpy as np
import jax
from functools import partial, wraps, reduce
from jax import numpy as jnp, jit, device_put
from typing import NamedTuple, Callable, Tuple, Any
from collections import namedtuple
from commplax import comm, xcomm, xop, adaptive_filter as af


class Layer(NamedTuple):
    name: str
    init: Callable
    apply: Callable
    vrange: Any


class LayerInitAns(NamedTuple):
    outshape: tuple
    weights: Any
    state: Any


class LayerApplyAns(NamedTuple):
    out: Any
    state: Any


class ValidRange(NamedTuple):
    begin: int
    end: int

    def __and__(self, other):
        if not isinstance(other, ValidRange):
            raise ValueError('expected ValidRange input but got {} instead'.format(type(other)))
        return ValidRange(self.begin + other.begin, self.end + other.end)

vrange0 = ValidRange(0, 0)

def make_vrangefn(vrange):
    def _join(vrin=vrange0):
        return vrin & vrange
    return _join


def layer(layer_maker, pure_fn=False, has_state=False):
    @wraps(layer_maker)
    def _layer_maker(*args, name=layer_maker.__name__, **kwargs):
        init, apply, vrange = (layer_maker(*args, **kwargs) + (vrange0,))[:3]

        if not callable(vrange):
            vrange = make_vrangefn(vrange)

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
            ret = ret[0], device_put(ret[1]), device_put(ret[2])
            return LayerInitAns(*ret)

        @partial(jit, static_argnums=3)
        @wraps(apply)
        def _apply_(weights, inputs, states, vrangein, **kwargs):
            if pure_fn:
                outputs = apply(inputs, vrangein, **kwargs)
            elif not has_state:
                outputs = apply(weights, inputs, vrangein, **kwargs)
            else:
                outputs, states = apply(weights, inputs, states, vrangein, **kwargs)
            return LayerApplyAns(outputs, states)

        def _apply(weights, inputs, states, vrangein=vrange0, **kwargs):
            return _apply_(weights, inputs, states, vrangein, **kwargs)

        return Layer(name, _init, _apply, vrange)

    return _layer_maker


fnlayer = partial(layer, pure_fn=True)
statlayer = partial(layer, has_state=True)


def grad(layer, has_aux=True, **kwargs):
    assert isinstance(layer, Layer)
    return jax.grad(layer.apply, has_aux=has_aux, **kwargs)


def value_and_grad(layer, has_aux=True, **kwargs):
    assert isinstance(layer, Layer)
    return jax.value_and_grad(layer.apply, has_aux=has_aux, **kwargs)


def _assert_tuple_len(t, l):
    assert isinstance(t, tuple) and len(t) == l


def _slice_valid(value, vrange):
    a = vrange[0]
    b = vrange[1]
    if value is not None: assert value.shape[0] > a - b
    value = value if value is None or a == 0 and b == 0 else value[a: b]
    return value


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


def _chained_call(fs, init, length=None):
    if callable(fs):
        fs = [fs] * length
    ret = init
    for f in fs:
        ret = f(ret)
    return ret


@fnlayer
def FOE(sps=2):
    def init(input_shape):
        return input_shape

    def apply(inputs, vrangein, *args, fo=None, **kwargs):
        vrangein = (vrangein[0] * sps, vrangein[1] * sps)
        fo = _slice_valid(fo, vrangein)
        return inputs * fo

    return init, apply


@layer
def DBP(sr, lspan, nspan, dtaps, lp, sps=2, vspan=None):
    steps = nspan if vspan is None else vspan
    n_invalid = (dtaps - 1) * steps
    vrange = ValidRange(n_invalid // 2 // sps, -n_invalid // 2 // sps)

    def init(input_shape):
        _, wD, wN = comm.dbp_params(sr, lspan, nspan, dtaps, launch_power=lp, virtual_spans=vspan)
        output_shape = (input_shape[0] - n_invalid,) + input_shape[1:]
        weights = (wD, wN * 0.2)
        return output_shape, weights

    def apply(weights, inputs, *args, **kwargs):
        wD, wN = weights
        outputs = xcomm.dbp_timedomain(inputs, wD, wN, mode='valid')
        return outputs

    return init, apply, vrange


@statlayer
def MIMO(taps=32, sps=2, train=False, mimo=af.ddlms, mimokwargs={}, mimoinitargs={}):
    mimo_init, mimo_update, mimo_apply = mimo(train=train, **mimokwargs)
    vrange = ValidRange(*(af.mimozerodelaypads(taps, sps)[0] // sps * np.array([1, -1])).tolist())

    def init(input_shape):
        dims = input_shape[-1]
        stats = mimo_init(dims=dims, taps=taps, **mimoinitargs)
        output_shape = (xop.frame_shape(input_shape, taps, sps, allowwaste=True)[0], dims)
        states = (0, stats)
        return output_shape, (), states

    def apply(weights, inputs, states, vrangein, *args, **kwargs):
        truth = _slice_valid(kwargs.get('truth'), vrange & vrangein)
        inputs = xop.frame(inputs, taps, sps)
        i, stats = states
        i, (stats, (params, _)) = af.iterate(mimo_update, i, stats, inputs, truth)
        outputs = mimo_apply(params, inputs)
        states = (i + inputs.shape[0],) + (stats,)
        return outputs, states

    return init, apply, vrange


@fnlayer
def Downsample(sps=2):
    def init(input_shape):
        return (input_shape[0] // sps,) + input_shape[1:]

    def apply(inputs, *args, **kwargs):
        return inputs[::sps]

    return init, apply


@fnlayer
def MSE():
    def init(input_shape):
        return ()

    def apply(inputs, vrangein, *args, truth=None, **kwargs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            y, x = inputs
        else:
            y, x = inputs, _slice_valid(truth, vrangein)
        return jnp.mean(jnp.abs(y - x)**2)

    return init, apply


@fnlayer
def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init = lambda input_shape: input_shape
    apply = lambda inputs, *args, **kwargs: fun(inputs, **fun_kwargs)
    return init, apply


Exp = elementwise(jnp.exp)


@fnlayer
def Identity():
    """Layer construction function for an identity layer."""
    init = lambda input_shape: input_shape
    apply = lambda inputs, *args, **kwargs: inputs
    return init, apply
Identity = Identity()


@fnlayer
def FanInElementwise(fun, **fun_kwargs):
    init = lambda input_shape: input_shape[0]
    apply = lambda inputs, *args, **kwargs: fun(*inputs, **fun_kwargs)
    vrange = lambda vranges: reduce(operator.and_, vranges)
    return init, apply, vrange


@fnlayer
def FanOut(num):
  """Layer construction function for a fan-out layer."""
  init = lambda input_shape: [input_shape] * num
  apply = lambda inputs, *args, **kwargs: [inputs] * num
  vrange = lambda vrangein=vrange0: (vrangein,) * num
  return init, apply, vrange


# Composing layers via combinators
@statlayer
def serial(*layers):
    """Combinator for composing layers in serial."""
    names, inits, applys, vranges = zip(*layers)
    nlayers = len(layers)
    names = _rename_dupnames(names)
    vrange = _chained_call(vranges, vrange0)

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

    def apply(weights, inputs, states, vrangein, **kwargs):
        new_states = []
        for fun, weight, state, vrange in zip(applys, weights, states, vranges):
            inputs, state = fun(weight, inputs, state, vrangein, **kwargs)
            vrangein = vrange(vrangein)
            new_states.append(state)
        return inputs, StateTuple(*new_states)

    return init, apply, vrange


@statlayer
def parallel(*layers):
    """Combinator for composing layers in parallel."""
    nlayers = len(layers)
    names, inits, applys, vranges = zip(*layers)
    names = _rename_dupnames(names)

    # group sublayers into namedtuple
    InputsTuple = _itp(names)
    WeightsTuple = _wtp(names)
    StateTuple = _stp(names)

    def init(input_shape):
        input_shapes, weights, states = tuple(zip(*[init(shape) for init, shape in zip(inits, input_shape)]))
        return InputsTuple(*input_shapes), WeightsTuple(*weights), StateTuple(*states)

    def apply(weights, inputs, states, vrangesin, **kwargs):
        assert len(vranges) == nlayers
        outputs, states = tuple(zip(*[f(w, x, s, vr, **kwargs) for f, w, x, s, vr in \
                           zip(applys, weights, inputs, states, vrangesin)]))
        return outputs, StateTuple(*states)

    def vrange(vrangesin):
        vrangeout = []
        for vr, vri in zip(vranges, vrangesin):
            vrangeout.append(vr(vri))
        return tuple(vrangeout)

    return init, apply, vrange



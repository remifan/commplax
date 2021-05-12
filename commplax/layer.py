import operator
import numpy as np
import jax
from functools import partial, wraps, reduce
from jax import numpy as jnp, jit, random, device_put, tree_util
from typing import NamedTuple, Callable, Tuple, Any
from commplax.util import namedtuple
from commplax import util, comm, xcomm, xop, adaptive_filter as af
from jax.nn.initializers import ones, zeros


class Layer(NamedTuple):
    name: str
    init: Callable
    apply: Callable
    truth_range: Any


class LayerInitAns(NamedTuple):
    outshape: tuple
    weights: Any
    state: Any


class LayerApplyAns(NamedTuple):
    out: Any
    state: Any


class TruthRange(NamedTuple):
    begin: int
    end: int

    def __and__(self, other):
        if not isinstance(other, TruthRange):
            raise ValueError('expected TruthRange input but got {} instead'.format(type(other)))
        return TruthRange(max([self.begin, other.begin]), min([self.end, other.end]))

    def __matmul__(self, other):
        if not isinstance(other, TruthRange):
            raise ValueError('expected TruthRange input but got {} instead'.format(type(other)))
        return TruthRange(self.begin + other.begin, self.end + other.end)

    def times(self, n: int):
        return TruthRange(self.begin * n, self.end * n)


TRANGE0 = TruthRange(0, 0)
EMPTY_WEIGHTS = ()
EMPTY_STATE = ()


def make_trangefn(trange):
    def _propogate(vrin=TRANGE0):
        return vrin @ trange
    return _propogate


def layer(layer_maker, pure_fn=False, has_state=False, name=None):
    @wraps(layer_maker)
    def _layer_maker(*args, name=layer_maker.__name__ if name is None else name, **kwargs):
        init, apply, trange = (layer_maker(*args, **kwargs) + (TRANGE0,))[:3]

        if not callable(trange):
            trange = make_trangefn(trange)

        @wraps(init)
        def _init(*args, **kwargs):
            ret = init(*args, **kwargs)
            if pure_fn:
                ret = (ret, EMPTY_WEIGHTS, EMPTY_STATE)
            elif not has_state:
                ret = ret + (EMPTY_STATE,)
            else:
                pass
            _assert_tuple_len(ret, 3)
            ret = ret[0], device_put(ret[1]), device_put(ret[2])
            return LayerInitAns(*ret)

        @partial(jit, static_argnums=3)
        @wraps(apply)
        def _apply_(weights, inputs, states, trangein, **kwargs):
            if pure_fn:
                outputs = apply(inputs, trangein, **kwargs)
            elif not has_state:
                outputs = apply(weights, inputs, trangein, **kwargs)
            else:
                outputs, states = apply(weights, inputs, states, trangein, **kwargs)
            return LayerApplyAns(outputs, states)

        def _apply(weights, inputs, states, trangein=TRANGE0, **kwargs):
            return _apply_(weights, inputs, states, trangein, **kwargs)

        return Layer(name, _init, _apply, trange)

    return _layer_maker


fnlayer = partial(layer, pure_fn=True)
statlayer = partial(layer, has_state=True)


def grad(layer, has_aux=True, **kwargs):
    assert isinstance(layer, Layer)
    return jax.grad(layer.apply, has_aux=has_aux, **kwargs)


def value_and_grad(layer, has_aux=True, **kwargs):
    assert isinstance(layer, Layer)
    return jax.value_and_grad(layer.apply, has_aux=has_aux, **kwargs)


def apply_trainable_grad(g_tree, b_tree):
    g_tree_def, b_tree_def = tree_util.tree_structure(g_tree), tree_util.tree_structure(b_tree)
    assert g_tree_def == b_tree_def, "gradients and trainable have mismatched tree structure"
    return tree_util.tree_map(lambda g_leaf, b_leaf: g_leaf if b_leaf else g_leaf * 0, g_tree, b_tree)


def trainable(weights, invert=False):
    return tree_util.tree_map(lambda _: not invert, weights)


def _assert_tuple_len(t, l):
    assert isinstance(t, tuple) and len(t) == l


def _slice_valid(value, trange):
    a = trange[0]
    b = trange[1]
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
    for n, i in zip(names, range(len(names))):
        if cnt[n] > 1:
            renames[i] = renames[i] + '0'
            cnt[n] = 0
    return renames


def _itp(names):
    ''' input_shapes namedtuple '''
    return namedtuple('InputShape', names, defaults=(None,) * len(names))


def _stp(names):
    ''' state namedtuple '''
    return namedtuple('State', names, defaults=(None,) * len(names))


def _wtp(names):
    ''' weights namedtuple '''
    return namedtuple('Weights', names, defaults=(None,) * len(names))


def _chained_call(fs, init, length=None):
    if callable(fs):
        fs = [fs] * length
    ret = init
    for f in fs:
        ret = f(ret)
    return ret


@layer
def Conv1d(taps=31, rtap=None, sps=1, mode='valid', winit=lambda k, s: np.zeros(s), dtype=jnp.complex64, conv=xop.fftconvolve):
    if not callable(winit):
        winit0 = winit
        taps = winit0.shape[0]
        winit = lambda *args: winit0
    if rtap is None:
        rtap = (taps - 1) // 2
    if mode == 'full':
        trange = (-rtap, taps - rtap - 1) #TODO: think more about this
        dlen = taps - 1
    elif mode == 'same':
        trange = (0, 0)
        dlen = 0
    elif mode == 'valid':
        trange = (rtap, rtap - taps + 1)
        dlen = 1 - taps
    else:
        raise ValueError('invalid mode {}'.format(mode))
    trange = TruthRange(trange[0] // sps, trange[1] // sps)

    def init(rng, input_shape):
        output_shape = (input_shape[0] + dlen,)
        weights = winit(rng, taps).astype(dtype)
        assert weights.shape[0] == taps
        assert output_shape[0] > 0
        return output_shape, weights

    def apply(weights, inputs, *args, **kwargs):
        return conv(inputs, weights, mode=mode)

    return init, apply, trange


@layer
def MIMOConv(taps=31, rtap=None, sps=1, mode='valid', winit=lambda k, s, d: np.zeros((s, d, d)), dtype=None, conv=xop.fftconvolve):
    if not callable(winit):
        winit0 = winit
        taps = winit0.shape[0]
        winit = lambda *args: winit0
    if rtap is None:
        rtap = (taps - 1) // 2
    if mode == 'full':
        trange = (-rtap, taps - rtap - 1) #TODO: think more about this
        dlen = taps - 1
    elif mode == 'same':
        trange = (0, 0)
        dlen = 0
    elif mode == 'valid':
        trange = (rtap, rtap - taps + 1)
        dlen = 1 - taps
    else:
        raise ValueError('invalid mode {}'.format(mode))

    trange = TruthRange(trange[0] // sps, trange[1] // sps)

    def init(rng, input_shape):
        dims = input_shape[1]
        output_shape = (input_shape[0] + dlen,) + input_shape[1:]
        weights = winit(rng, taps, dims)
        if dtype is not None:
            weights = weights.astype(dtype)
        assert weights.shape[0] == taps
        assert output_shape[0] > 0
        return output_shape, weights

    def apply(weights, inputs, *args, **kwargs):
        return xcomm.mimoconv(inputs, weights, mode=mode, conv=conv)

    return init, apply, trange


@fnlayer
def FOE(sps=2):
    def init(rng, input_shape):
        return input_shape

    def apply(inputs, trangein, *args, fo=None, **kwargs):
        trangein = (trangein[0] * sps, trangein[1] * sps)
        fo = _slice_valid(fo, trangein)
        return inputs * fo

    return init, apply


# @layer
# def DBP(sr, lspan, nspan, dtaps, lp, sps=2, vspan=None):
#     steps = nspan if vspan is None else vspan
#     n_invalid = (dtaps - 1) * steps
#     trange = TruthRange(n_invalid // 2 // sps, -n_invalid // 2 // sps)

#     def init(input_shape):
#         _, wD, wN = comm.dbp_params(sr, lspan, nspan, dtaps, launch_power=lp, virtual_spans=vspan)
#         output_shape = (input_shape[0] - n_invalid,) + input_shape[1:]
#         weights = (wD, wN * 0.2)
#         return output_shape, weights

#     def apply(weights, inputs, *args, **kwargs):
#         wD, wN = weights
#         outputs = xcomm.dbp_timedomain(inputs, wD, wN, mode='valid')
#         return outputs

#     return init, apply, trange


@statlayer
def MIMOAEq(taps=32, sps=2, train=False, mimo=af.ddlms, mimokwargs={}, mimoinitargs={}):
    mimo_init, mimo_update, mimo_apply = mimo(train=train, **mimokwargs)
    trange = TruthRange(*(af.mimozerodelaypads(taps, sps)[0] // sps * np.array([1, -1])).tolist())

    def init(rng, input_shape):
        dims = input_shape[-1]
        stats = mimo_init(dims=dims, taps=taps, **mimoinitargs)
        output_shape = (xop.frame_shape(input_shape, taps, sps, allowwaste=True)[0], dims)
        states = (0, stats)
        return output_shape, (), states

    def apply(weights, inputs, states, trangein, *args, **kwargs):
        truth = _slice_valid(kwargs.get('truth'), trange @ trangein)
        inputs = xop.frame(inputs, taps, sps)
        i, stats = states
        i, (stats, (params, _)) = af.iterate(mimo_update, i, stats, inputs, truth)
        outputs = mimo_apply(params, inputs)
        states = (i + inputs.shape[0],) + (stats,)
        return outputs, states

    return init, apply, trange


@fnlayer
def Slice(s):
    def init(rng, input_shape):
        return (input_shape[0] - s[0] + s[1] if s[1] <= 0 else s[1] - s[0],) + input_shape[1:]
    def apply(inputs, *args, **kwargs):
        return inputs[s[0]: s[1] if s[1] < 0 else inputs.shape[0] - s[1], ...]
    return init, apply


@fnlayer
def MSE():
    def init(rng, input_shape):
        return ()

    def apply(inputs, trangein, *args, truth=None, **kwargs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            y, x = inputs
        else:
            y, x = inputs, _slice_valid(truth, trangein)
        return jnp.mean(jnp.abs(y - x)**2)

    return init, apply


@layer
def BatchNorm(axis=(0,), epsilon=1e-5, center=False, scale=False,
              beta_init=zeros, gamma_init=ones, normalizer=jax.nn.normalize):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
    axis = (axis,) if jnp.isscalar(axis) else axis
    def init(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
        k1, k2 = random.split(rng)
        beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)
        return input_shape, (beta, gamma)
    def apply(weights, x, *args, **kwargs):
        beta, gamma = weights
        ed = tuple(None if i in axis else slice(None) for i in range(jnp.ndim(x)))
        z = normalizer(x, axis, epsilon=epsilon)
        if center and scale: return gamma[ed] * z + beta[ed]
        if center: return z + beta[ed]
        if scale: return gamma[ed] * z
        return z
    return init, apply


@fnlayer
def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init = lambda rng, input_shape: input_shape
    apply = lambda inputs, *args, **kwargs: fun(inputs, **fun_kwargs)
    return init, apply


@layer
def ElementwiseFn(fun, winit=lambda *args: (), **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init = lambda rng, input_shape: (input_shape, winit(input_shape))
    apply = lambda weights, inputs, *args, **kwargs: fun(weights, inputs, **fun_kwargs)
    return init, apply


@fnlayer
def Identity():
    """Layer construction function for an identity layer."""
    init = lambda rng, input_shape: input_shape
    apply = lambda inputs, *args, **kwargs: inputs
    return init, apply
Identity = Identity()


@fnlayer
def FanInStack(axis=-1):
    def init(rng, input_shape):
        ax = axis % (len(input_shape[0]) + 1)
        output_shape = input_shape[0][:ax] + (len(input_shape),) + input_shape[0][ax:]
        return output_shape
    def apply(inputs, *args, **kwargs):
        return jnp.stack(inputs, axis=axis)
    trange = lambda tranges: reduce(operator.and_, tranges)
    return init, apply, trange


@fnlayer
def FanInElementwise(fun, **fun_kwargs):
    init = lambda rng, input_shape: input_shape[0]
    apply = lambda inputs, *args, **kwargs: fun(*inputs, **fun_kwargs)
    trange = lambda tranges: reduce(operator.and_, tranges)
    return init, apply, trange


@fnlayer
def FanOut(num):
  """Layer construction function for a fan-out layer."""
  init = lambda rng, input_shape: [input_shape] * num
  apply = lambda inputs, *args, **kwargs: [inputs] * num
  trange = lambda trangein=TRANGE0: (trangein,) * num
  return init, apply, trange


@fnlayer
def FanOutAxis(axis=-1):
    def init(rng, input_shape):
        ax = axis % len(input_shape)
        output_shape = []
        for _ in range(input_shape[ax]):
            output_shape.append(input_shape[:ax] + input_shape[ax+1:])
        return output_shape

    def apply(inputs, *args, **kwargs):
        outputs = []
        for inp in np.moveaxis(inputs, axis, 0):
            outputs.append(inp)
        return outputs

    trange = lambda trangein=TRANGE0: (trangein,)
    return init, apply, trange


# Composing layers via combinators
def serial(*layers):
    """Combinator for composing layers in serial."""
    names, inits, applys, tranges = zip(*layers)
    nlayers = len(layers)
    names = _rename_dupnames(names)
    trange = _chained_call(tranges, TRANGE0)

    # group sublayers into namedtuple
    WeightsTuple = _wtp(names)
    StateTuple = _stp(names)

    def init(rng, input_shape):
        weights = []
        states = []
        for init in inits:
            rng, layer_rng = random.split(rng)
            input_shape, weight, state = init(layer_rng, input_shape)
            weights.append(weight)
            states.append(state)
        return input_shape, WeightsTuple(*weights), StateTuple(*states)

    def apply(weights, inputs, states, trangein, **kwargs):
        assert weights._fields == WeightsTuple()._fields, "weights mismatch"
        assert states._fields == StateTuple()._fields, "state mismatch"
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        new_states = []
        for fun, weight, state, trange, rng in zip(applys, weights, states, tranges, rngs):
            inputs, state = fun(weight, inputs, state, trangein, rng=rng, **kwargs)
            trangein = trange(trangein)
            new_states.append(state)
        return inputs, StateTuple(*new_states)

    return init, apply, trange
s = statlayer(serial, name='s') # short alias
serial = statlayer(serial, name='serial')


def parallel(*layers):
    """Combinator for composing layers in parallel."""
    nlayers = len(layers)
    names, inits, applys, tranges = zip(*layers)
    names = _rename_dupnames(names)

    # group sublayers into namedtuple
    InputsTuple = _itp(names)
    WeightsTuple = _wtp(names)
    StateTuple = _stp(names)

    def init(rng, input_shape):
        rngs = random.split(rng, nlayers)
        input_shapes, weights, states = tuple(zip(*[init(rng, shape) for init, rng, shape in zip(inits, rngs, input_shape)]))
        return InputsTuple(*input_shapes), WeightsTuple(*weights), StateTuple(*states)

    def apply(weights, inputs, states, trangesin, **kwargs):
        if len(trangesin) == 1: trangesin = trangesin * nlayers # broadcast for input_shape dependent layer e.g. FanOutAxis
        assert len(trangesin) == nlayers
        assert weights._fields == WeightsTuple()._fields, "weights mismatch"
        assert states._fields == StateTuple()._fields, "state mismatch"
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        outputs, states = tuple(zip(*[f(w, x, s, vr, rng=r, **kwargs) for f, w, x, s, vr, r in \
                           zip(applys, weights, inputs, states, trangesin, rngs)]))
        return outputs, StateTuple(*states)

    def trange(trangesin):
        trangeout = []
        for vr, vri in zip(tranges, trangesin):
            trangeout.append(vr(vri))
        return tuple(trangeout)

    return init, apply, trange
p = statlayer(parallel, name='p') # short alias
parallel = statlayer(parallel, name='parallel')


def shape_dependent(make_layer):
  name = make_layer().name
  def init(rng, input_shape, *args, **kwargs):
    return make_layer(input_shape).init(rng, input_shape, *args, **kwargs)
  def apply(weights, inputs, states, trangein,  **kwargs):
    return make_layer(inputs.shape).apply(weights, inputs, states, trangein, **kwargs)
  def trange(trangein):
    return make_layer().trange(trangein)
  return name, init, apply, trange



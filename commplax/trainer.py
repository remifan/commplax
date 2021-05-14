from collections import namedtuple
from jax import tree_util, jit, random
from functools import partial
from commplax import layer as cl, cxopt
from tqdm.notebook import tqdm, trange
import numpy as np


def apply_trainable_grad(g_tree, b_tree):
    g_tree_def, b_tree_def = tree_util.tree_structure(g_tree), tree_util.tree_structure(b_tree)
    assert g_tree_def == b_tree_def, "gradients and trainable have mismatched tree structure"
    return tree_util.tree_map(lambda g_leaf, b_leaf: g_leaf if b_leaf else g_leaf * 0, g_tree, b_tree)


def get_trainable(weights, invert=False):
    return tree_util.tree_map(lambda _: not invert, weights)


# TODO: refine with class and type checking
# noun (1st-order function)
_all = lambda t: t

# verb (2nd-order function)
_train = lambda n: lambda t: t.apply(n(t), True)
_freeze = lambda n: lambda t: t.apply(n(t), False)
_train.T = _freeze
_freeze.T = _train

# decorator (3rd-order function)
_only = lambda v: lambda n: lambda t: v(n)(v.T(_all)(t))

# group all maps
TrainableMap = namedtuple('TrainableMap', 'all train freeze only')
tmap = TrainableMap(all=_all,
                    train=_train,
                    freeze=_freeze,
                    only=_only)


TrainResult = namedtuple('TrainResult', 'weights aux')


def train(model: cl.Layer,
          batch_generator,
          loss_metric: cl.Layer=cl.MSE(),
          input_shape=(None, 2),
          trainable=tmap.all,
          optimizer=cxopt.adam(1e-5),
          rng=random.PRNGKey(0),
          s_batch=500,
          n_iter=None,
          n_epoch=1,
          jit_backend='cpu',
          stat_man=lambda i, s: s):
    assert isinstance(model, cl.Layer)
    assert isinstance(loss_metric, cl.Layer)

    model_loss = cl.serial(
        model,
        loss_metric,
    )

    weights, model_state0 = model_loss.init(rng, (50000, input_shape[1]))[1:]
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(weights)
    w_trainable = trainable(get_trainable(weights))

    @partial(jit, backend=jit_backend)
    def step(i, opt_state: cxopt.OptimizerState, model_state, batch):
        inputs, targets, aux = (*batch, ())[:3]
        kwargs = {} if len(aux) == 0 else aux._asdict()
        weights = get_params(opt_state)
        (loss, model_state), g = cl.value_and_grad(model_loss)(weights, inputs, model_state, truth=targets, **kwargs)
        opt_state = opt_update(i, apply_trainable_grad(g, w_trainable), opt_state)
        return loss, g, model_state, opt_state

    batch_gens = []
    batch_num = 0
    for _ in range(n_epoch):
        batch_num, batch_gen = batch_generator(model_loss, s_batch)
        batch_gens.append(batch_gen)
    assert batch_num > 0

    AuxOut = namedtuple('AuxOut', 'loss grad state')

    n_iter = n_epoch * batch_num if n_iter is None else n_iter

    loss = None
    grad = None
    i_iter = 0
    for i_epoch in range(n_epoch):
        model_state = model_state0
        for _ in trange(min(n_iter, batch_num), desc='epoch %d/%d' % (i_epoch, n_epoch), leave=False):
            ind = (i_iter, i_epoch)
            loss, grad, model_state, opt_state = step(i_iter,
                                                      opt_state,
                                                      stat_man(ind, model_state),
                                                      next(batch_gens[i_epoch]))
            i_iter += 1
            n_iter -= 1
            yield TrainResult(get_params(opt_state), AuxOut(float(loss), grad, model_state))



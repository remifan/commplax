from jax import tree_util, jit, random, numpy as jnp
from functools import partial
from commplax import layer as cl, cxopt, util
from tqdm.notebook import tqdm, trange
from collections import namedtuple


def apply_trainable_grad(g_tree, t_tree, mask=True):
    g_tree_def, t_tree_def = tree_util.tree_structure(g_tree), tree_util.tree_structure(t_tree)
    assert g_tree_def == t_tree_def, "gradients and trainable have mismatched tree structure"
    return tree_util.tree_map(lambda g, t: g * (t & mask), g_tree, t_tree)


def assertvar(v1, v2, msg="mismatched variables"):
    v1_tree_def = tree_util.tree_structure(v1)
    v2_tree_def = tree_util.tree_structure(v2)
    assert v1_tree_def == v2_tree_def, "{} given\n {} \n {}".format(msg, v1_tree_def, v2_tree_def)


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
TrainableMap = namedtuple('TrainableMap', 'all only train freeze train_all freeze_all only_train only_freeze')
tmap = TrainableMap(all=_all,
                    only=_only,
                    train=_train,
                    freeze=_freeze,
                    train_all=_train(_all),
                    freeze_all=_freeze(_all),
                    only_train=_only(_train),
                    only_freeze=_only(_freeze))


def supervised(model: cl.Layer,
               batch_generator,
               loss_metric: cl.Layer=cl.MSE(),
               input_shape=(None, 2),
               init_weights=None,
               init_state=None,
               trainable=tmap.train_all,
               optimizer=cxopt.adam(1e-5),
               freeze_weights=False,
               rng=random.PRNGKey(0),
               s_batch=500,
               n_iter=None,
               n_epoch=1,
               jit_backend='cpu',
               tqdm_kwargs={},
               state_schedule=lambda i, s0, s: s):

    assert isinstance(model, cl.Layer), "invalid model"
    assert isinstance(loss_metric, cl.Layer), "invalid loss"

    model_loss = cl.serial(
        model,
        loss_metric,
    )

    weights0, state0 = model_loss.init(rng, (50000, input_shape[1]))[1:]
    weights0 = weights0 if init_weights is None else init_weights
    state0 = state0 if init_state is None else init_state
    # assertvar(weights.model, weights0.model, "invalid initial model weights") # too strict
    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(weights0)
    weights = get_params(opt_state)
    state = state0
    w_trainable = trainable(util.tree_full(weights, True))

    TrainResult = namedtuple('TrainResult', 'weights aux')
    AuxOut = namedtuple('AuxOut', 'loss grad state')

    if util.tree_any(w_trainable):
        batch_num = 0
        batch_gens = []
        for _ in range(n_epoch):
            batch_num, batch_gen = batch_generator(model, s_batch)
            batch_gens.append(batch_gen)
        assert batch_num > 0, "empty batch"

        n_iter = n_epoch * batch_num if n_iter is None else n_iter
        freeze_weights = cxopt.make_schedule(freeze_weights)

        @partial(jit, backend=jit_backend)
        def step(i, opt_state: cxopt.OptimizerState, state, batch):
            inputs, targets, aux = (*batch, ())[:3]
            kwargs = {} if len(aux) == 0 else aux._asdict()
            weights = get_params(opt_state)
            (loss, state), g = cl.value_and_grad(model_loss)(weights, inputs, state, truth=targets, **kwargs)
            opt_state = opt_update(i, apply_trainable_grad(g, w_trainable, not freeze_weights(i)), opt_state)
            return loss, g, state, opt_state

        loss = None
        grad = None
        i_iter = 0
        for i_epoch in range(n_epoch):
            state = state_schedule(i_epoch, state0, state)
            for _ in trange(min(n_iter, batch_num), **util.passkwargs(tqdm_kwargs, desc='epoch %d/%d' % (i_epoch, n_epoch))):
                loss, grad, state, opt_state = step(i_iter,
                                                          opt_state,
                                                          state,
                                                          next(batch_gens[i_epoch]))
                i_iter += 1
                n_iter -= 1
                yield TrainResult(get_params(opt_state), AuxOut(float(loss), grad, state))
    else:
        yield TrainResult(weights, AuxOut(None, None, state))


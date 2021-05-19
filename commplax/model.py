from typing import NamedTuple, Any
from commplax import layer as cl
from jax import random


class Model(NamedTuple):
    model: cl.Layer
    hparams: Any
    state: Any
    weights: Any



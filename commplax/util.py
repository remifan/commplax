import jax

def gpuexists():
    gpus = jax.devices('gpu')
    return len(gpus) != 0

def gpufirstbackend():
    '''
    `backend` api is experimental feature,
    https://jax.readthedocs.io/en/latest/jax.html#jax.jit
    '''
    return 'gpu' if gpuexists() else 'cpu'


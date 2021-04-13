import jax

def devputlike(x, y):
    '''put x into the same device with y'''
    return jax.device_put(x, y.device_buffer.device())


def gpuexists():
    gpus = jax.devices('gpu')
    return len(gpus) != 0


def gpufirstbackend():
    '''
    NOTE: `backend` api is experimental feature,
    https://jax.readthedocs.io/en/latest/jax.html#jax.jit
    '''
    return 'gpu' if gpuexists() else 'cpu'


import jax
import numpy as np
from jax import numpy as jnp, jit, vmap
from functools import partial
from collections import namedtuple
from typing import NamedTuple, Any, Optional, Union, Callable
from commplax import xop, comm, xcomm, adaptive_filter as af, util


class LMSHparams(NamedTuple):
    taps: int = 31
    lr_w: float = 1/2**6
    lr_f: float = 1/2**7
    lr_b: float = 1/2**11
    lockgain: bool = False
    train: Any = True


class DBPHparams(NamedTuple):
    sr: Optional[float] = None
    lspan: Optional[int] = None
    nspan: Optional[int] = None
    dtaps: Optional[int] = None
    vspan: Optional[int] = None
    xi: float = .5
    lpdbm: float = 30.
    stps: int = 1
    fc: float = 299792458/1550E-9
    disp: float = 16.5E-6
    ntaps: int=1
    nmimo: bool = False


class MFHparams(NamedTuple):
    taps: int = 129
    kind: str = 'delta'


class DBPParams(NamedTuple):
    d: Optional[Union[jnp.ndarray, np.ndarray]] = None
    n: Optional[Union[jnp.ndarray, np.ndarray]] = None


class DBPLMSParams(NamedTuple):
    mf: Optional[Union[jnp.ndarray, np.ndarray]] = None
    dbp: DBPParams = DBPParams()


class DBPLMSInput(NamedTuple):
    y: Optional[Union[jnp.ndarray, np.ndarray]]
    x: Optional[Union[jnp.ndarray, np.ndarray]]
    f: Optional[Union[jnp.ndarray, np.ndarray]]


DBPLMSInpBuf = namedtuple('DBPLMSInpBuf', ['ybuf', 'xbuf', 'fomulbuf'])
DBPLMSAFStat = namedtuple('DBPLMSAFStat', ['mimo'])
DBPLMSAFVals = namedtuple('DBPLMSAFVals', ['mimo'])


class DSPModel(NamedTuple):
    params: DBPLMSParams
    inpbuf: DBPLMSInpBuf
    afstat: DBPLMSAFStat
    delay: int
    eziter: Callable
    iterate: Callable
    model: Callable
    grad: Callable
    value_and_grad: Callable


def matchedfilter(y, h, mode='valid'):
    return vmap(lambda y, h: xop.convolve(y, h, mode=mode), in_axes=-1, out_axes=-1)(y, h)


def delta(taps, dims=2, dtype=np.complex64):
    mf = np.zeros((taps, dims), dtype=dtype)
    mf[(taps - 1) // 2, :] = 1.
    return mf


def dbpinit(hparams: DBPHparams, value: DBPParams=DBPParams(), dims=2):
    if value is None:
        value = DBPParams()
    _, paramD, paramN = comm.dbp_params(hparams.sr,
                                        hparams.lspan,
                                        hparams.nspan,
                                        hparams.dtaps,
                                        launch_power=hparams.lpdbm,
                                        steps_per_span=hparams.stps,
                                        virtual_spans=hparams.vspan,
                                        fiber_dispersion=hparams.disp)
    if hparams.ntaps > 1:
        # extend to mimo filter
        paramN = np.stack([delta(hparams.ntaps, dims=dims * dims, dtype=np.float64).reshape((hparams.ntaps, dims, dims))] \
                          * paramN.shape[0]) * paramN[:, None, :, :]
    defvalue = DBPParams(d=paramD, n=paramN * hparams.xi)
    # allow partial update
    value = DBPParams(d=defvalue.d if value.d is None else value.d,
                      n=defvalue.n if value.n is None else value.n)
    steps = value.d.shape[0]
    dtaps = value.d.shape[1]
    ntaps = value.n.shape[1] if value.n.ndim > 3 else 1
    taps = steps * (dtaps - 1 + ntaps - 1) + 1 # equivalent linear filter taps
    valid = af.filterzerodelaypads(taps)[0]
    return value, valid


def mfinit(hparams: MFHparams, value=None, dims=2):
    if value is None:
        if hparams.kind.lower() == 'delta':
            value = delta(hparams.taps, dims)
        else:
            raise ValueError('matched filter of the specified kind is not support yet')

    taps = value.shape[0]
    valid = af.filterzerodelaypads(taps)[0]
    return value, valid


def dbplms(var: DBPLMSParams=DBPLMSParams(),
           sps: int=2,
           dims: int=2,
           modformat: str='16QAM',
           dbphparams: DBPHparams=DBPHparams(),
           mfhparams: MFHparams=MFHparams(),
           lmshparams: LMSHparams=LMSHparams()):

    qamscale = xcomm.qamscale(modformat)

    lms_init, lms_update, lms_map = af.ddlms(lr_w=lmshparams.lr_w,
                                             lr_f=lmshparams.lr_f,
                                             lr_b=lmshparams.lr_b,
                                             train=lmshparams.train,
                                             lockgain=lmshparams.lockgain)
    lms_vld = af.filterzerodelaypads(lmshparams.taps, stride=sps)[0]
    dbp, dbp_vld = dbpinit(dbphparams, value=var.dbp)
    mf, mf_vld = mfinit(mfhparams, value=var.mf)

    ybuf = jnp.zeros(((dbp_vld + mf_vld + lms_vld).sum(), dims), dtype=np.complex64)
    xbuf = jnp.zeros(((dbp_vld[1] + mf_vld[1] + lms_vld[1]) // sps, dims), dtype=np.complex64)
    fobuf = jnp.ones((ybuf.shape[0], 1), dtype=np.complex64)
    dspdelay = xbuf.shape[0]

    params0 = DBPLMSParams(mf, dbp)
    inpbuf0 = DBPLMSInpBuf(ybuf, xbuf, fobuf)
    afstate0 = DBPLMSAFStat(lms_init(lmshparams.taps, mimoinit='centralspike'))
    step0 = 0

    @jit
    def streaminputs(inp: tuple, inpbuf: DBPLMSInpBuf):
        y, x, fomul = inp
        ybuf, xbuf, fobuf, = inpbuf
        y = jnp.concatenate([ybuf, y])
        ybuf = y[-ybuf.shape[0]:]
        x = jnp.concatenate([xbuf, x])
        xbuf = x[-xbuf.shape[0]:]
        x = x[:-xbuf.shape[0]]
        fomul = jnp.concatenate([fobuf, fomul])
        fobuf = fomul[-fobuf.shape[0]:]
        fomul = fomul[dbp_vld[0]:-dbp_vld[1]]
        return DBPLMSInput(y, x, fomul), DBPLMSInpBuf(ybuf, xbuf, fobuf)

    def tddbp(y, d, n, conv=xop.fftconvolve):
        return xcomm.dbp_timedomain(y, d, n, mode='valid', homosteps=True, scansteps=True, conv=conv, mimolpf=dbphparams.nmimo)

    def iterate(step, inp: tuple, params: DBPLMSParams, inpbuf: DBPLMSInpBuf, afstate: DBPLMSAFStat, conv=xop.fftconvolve):
        inp, inpbuf = streaminputs(inp, inpbuf)
        params = util.tree_update(params0, params)
        z = tddbp(inp.y, params.dbp.d, params.dbp.n, conv=conv)
        z *= inp.f
        z = matchedfilter(z, params.mf, mode='valid')
        zf = xop.frame(z, lmshparams.taps, sps)
        step, (mimostat, (mimovals, _)) = af.iterate(step, lms_update, afstate.mimo, zf, inp.x)
        z = lms_map(mimovals, zf)
        return step, z, inpbuf, DBPLMSAFStat(mimo=mimostat), DBPLMSAFVals(mimo=mimovals)

    def model(params: DBPLMSParams, inp: tuple, inpbuf: DBPLMSInpBuf, afvals: DBPLMSAFVals, conv=xop.fftconvolve):
        inp, inpbuf = streaminputs(inp, inpbuf)
        z = tddbp(inp.y, params.dbp.d, params.dbp.n, conv=conv)
        z *= inp.f
        z = matchedfilter(z, params.mf, mode='valid')
        zf = xop.frame(z, lmshparams.taps, sps)
        z = lms_map(afvals.mimo, zf)
        return z, inpbuf

    def lossfn(v: DBPLMSParams, z: Any, inp: tuple, params: DBPLMSParams, inpbuf: DBPLMSInpBuf, afvals: DBPLMSAFVals):
        zhat, _ = model(util.tree_update(params, v), inp, inpbuf, afvals)
        return jnp.mean(jnp.abs(zhat - z)**2)

    def value_and_grad(v: DBPLMSParams, z: jnp.ndarray, inp: tuple, inpbuf: DBPLMSInpBuf, afvals: DBPLMSAFVals, params: DBPLMSParams=params0):
        loss, grads = jax.value_and_grad(lossfn)(v, z, inp, params, inpbuf, afvals)
        return loss, grads

    def grad(v: DBPLMSParams, z: jnp.ndarray, inp: tuple, inpbuf: DBPLMSInpBuf, afvals: DBPLMSAFVals, params: DBPLMSParams=params0):
        loss, grads = jax.grad(lossfn)(v, z, inp, params, inpbuf, afvals)
        return loss, grads

    def eziter(inp: tuple, params: DBPLMSParams, zerodelayroll=True, backend='cpu', conv=xop.conv1d_fft_oa):
        z = jit(partial(iterate, conv=conv), backend=backend)(step0, inp, params, inpbuf0, afstate0)[1]
        return xop.delay(z, -dspdelay) if zerodelayroll else z

    return DSPModel(params0, inpbuf0, afstate0, dspdelay, eziter, iterate, model, grad, value_and_grad)



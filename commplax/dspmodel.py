import jax
import numpy as np
from jax import numpy as jnp, jit, vmap
from functools import partial
from collections import namedtuple
from commplax import op, xop, comm, xcomm, cxopt, adaptive_filter as af, equalizer as eq


DDLMSParams = namedtuple('LMSMIMOParams', ['taps', 'lr_w', 'lr_f', 'lockgain'], defaults=(31, 1/2**7, 1/2**7, False))
DBPParams = namedtuple('DBPParams', ['h', 'c'])


def matchedfilter(y, h, mode='same'):
  return vmap(lambda y, h: xop.convolve(y, h, mode=mode), in_axes=-1, out_axes=-1)(y, h)


def onespike(taps, dims=2):
    mf = np.zeros((taps, dims), dtype=np.complex64)
    mf[(taps - 1) // 2, :] = 1.
    return jnp.array(mf)


def ddlms(a,
          mf=onespike(129),
          dbpparams=None,
          lmsparams=(DDLMSParams(taps=11, lr_w=1/2**7, lr_f=1/2**7, lockgain=False))):
    lmsparams = lmsparams
    mf = mf
    sps = 2
    dims = 2
    lms_init, lms_update, lms_map = af.ddlms(lr_w=lmsparams.lr_w,
                                             lr_f=lmsparams.lr_f,
                                             lockgain=lmsparams.lockgain)

    mftaps = mf.shape[0]
    mimotaps = lmsparams.taps
    if dbpparams is not None:
        cdctaps = dbpparams.h.shape[0] * (dbpparams.h.shape[1] - 1) + 1
    else:
        cdctaps = eq.cdctaps(a['samplerate'], a['cd'])
    cdcpads = af.filterzerodelaypads(cdctaps)
    mfpads = af.filterzerodelaypads(mftaps)
    mimopads = af.filterzerodelaypads(mimotaps, stride=sps)
    ybuf = jnp.zeros(((cdcpads[0] + mfpads[0] + mimopads[0]).sum(), dims), dtype=np.complex64)
    xbuf = jnp.zeros(((cdcpads[0, 0] + mfpads[0, 0] + mimopads[0, 0]) // sps, dims), dtype=np.complex64)
    fobuf = jnp.ones((ybuf.shape[0] - cdcpads[0, 0], 1), dtype=np.complex64)
    mimostat = lms_init(mimotaps, mimoinit='centralspike')
    dspdelay = xbuf.shape[0]

    Params = namedtuple('Params', ['mf', 'dbp'])
    InpBuf = namedtuple('InpBuf', ['ybuf', 'xbuf', 'fomulbuf'])
    AFStat = namedtuple('AFStat', ['mimo'])
    AFVals = namedtuple('AFVals', ['mimo'])

    params0 = Params(mf, dbpparams)
    inpbuf0 = InpBuf(ybuf, xbuf, fobuf)
    afstate0 = AFStat(mimostat)

    @jit
    def updinpbuf(inp, inpbuf):
        y, x, fomul = inp
        ybuf, xbuf, fobuf, = inpbuf
        y = jnp.concatenate([ybuf, y])
        ybuf = y[-ybuf.shape[0]:]
        x = jnp.concatenate([xbuf, x])
        xbuf = x[-xbuf.shape[0]:]
        x = x[:-xbuf.shape[0]]
        fomul = jnp.concatenate([fobuf, fomul])
        fobuf = fomul[-fobuf.shape[0]:]
        fomul = fomul[:-cdcpads[0, 1]]
        return (y, x, fomul), InpBuf(ybuf, xbuf, fobuf)

    @jit
    def iterate(inp, params, inpbuf, afstate):
        inp, inpbuf = updinpbuf(inp, inpbuf)

        y, x, fomul = inp
        mf, dbp = params
        mimostat, = afstate

        if dbpparams is not None:
            z = xcomm.dbp_timedomain(y / jnp.sqrt(2), dbp.h, dbp.c, mode='valid', homosteps=True, scansteps=True) * jnp.sqrt(2)
        else:
            z = eq.cdcomp(y, a['samplerate'], a['cd'], mode='valid')
        z *= fomul
        z = matchedfilter(z, mf, mode='valid')
        zf = xop.frame(z, lmsparams.taps, sps)
        mimostat, (mimovals, _) = af.iterate(lms_update, mimostat, zf, x)
        z = lms_map(mimovals, zf)
        z *= xcomm.qamscale(a['modformat'])

        afstate = AFStat(mimostat,)
        afvals = AFVals(mimovals,)
        return z, inpbuf, afstate, afvals

    @jit
    def model(inp, params, inpbuf, afvals):
        inp, inpbuf = updinpbuf(inp, inpbuf)

        y, _, fomul = inp
        mf, dbp = params
        mimovals, = afvals

        if dbpparams is not None:
            z = xcomm.dbp_timedomain(y / jnp.sqrt(2), dbp.h, dbp.c, mode='valid', homosteps=True, scansteps=True) * jnp.sqrt(2)
        else:
            z = eq.cdcomp(y, a['samplerate'], a['cd'], mode='valid')
        z *= fomul
        z = matchedfilter(z, mf, mode='valid')
        zf = xop.frame(z, lmsparams.taps, sps)
        z = lms_map(mimovals, zf)
        z *= xcomm.qamscale(a['modformat'])
        return z, inpbuf

    return iterate, model, dspdelay, params0, inpbuf0, afstate0



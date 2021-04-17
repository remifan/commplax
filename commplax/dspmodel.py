import jax
import numpy as np
from jax import numpy as jnp, jit, vmap
from functools import partial
from collections import namedtuple
from commplax import op, xop, comm, xcomm, adaptive_filter as af, equalizer as eq


DSPModel = namedtuple('DSPModel', ['params', 'inpbuf', 'afstat', 'delay', 'eziter', 'iter', 'model', 'value_and_grad'])
LMSHparams = namedtuple('LMSHparams', ['taps', 'lr_w', 'lr_f', 'lockgain'], defaults=(31, 1/2**6, 1/2**7, False))
DBPHparams = namedtuple('DBPHparams', ['sr', 'lspan', 'nspan', 'taps', 'xi', 'lpdbm', 'stps', 'vspan', 'fc', 'disp'],
                       defaults=(None, None, None, None, 0.5, 30., 1, None, 299792458/1550E-9, 16.5E-6))
CDCHparams = namedtuple('CDCHparams', ['sr', 'cd'])
MFHparams = namedtuple('MFHparams', ['taps', 'kind'], defaults=(129, 'delta'))
DBPParams = namedtuple('DBPParams', ['d', 'n'])


def matchedfilter(y, h, mode='same'):
  return vmap(lambda y, h: xop.convolve(y, h, mode=mode), in_axes=-1, out_axes=-1)(y, h)


def delta(taps, dims=2):
    mf = np.zeros((taps, dims), dtype=np.complex64)
    mf[(taps - 1) // 2, :] = 1.
    return jnp.array(mf)


def dbplms(sps=2,
           dims=2,
           modformat='16QAM',
           dbphparams: DBPHparams=None,
           cdchparams: CDCHparams=None,
           mfhparams: MFHparams=MFHparams(taps=129, kind='delta'),
           lmshparams: LMSHparams=(LMSHparams(taps=11, lr_w=1/2**7, lr_f=1/2**7, lockgain=False))):

    lms_init, lms_update, lms_map = af.ddlms(lr_w=lmshparams.lr_w,
                                             lr_f=lmshparams.lr_f,
                                             lockgain=lmshparams.lockgain)

    if mfhparams.kind.lower() == 'delta':
        mf = delta(mfhparams.taps)
    else:
        raise ValueError('matched filter of the specified kind is not support yet')
    mftaps = mf.shape[0]
    mimotaps = lmshparams.taps

    qamscale = xcomm.qamscale(modformat)

    if dbphparams is not None:
        _, paramD, paramN = comm.dbp_params(dbphparams.sr,
                                            dbphparams.lspan,
                                            dbphparams.nspan,
                                            dbphparams.taps,
                                            launch_power=dbphparams.lpdbm,
                                            steps_per_span=dbphparams.stps,
                                            virtual_spans=dbphparams.vspan,
                                            fiber_dispersion=dbphparams.disp)
        dbp = DBPParams(d=paramD, n=paramN * dbphparams.xi)
        cdctaps = dbp.d.shape[0] * (dbp.d.shape[1] - 1) + 1 # equivalent taps
    elif cdchparams is not None:
        dbp = DBPParams(d=None, n=None)
        cdctaps = eq.cdctaps(cdchparams.sr, cdchparams.cd)
    else:
        raise ValueError('dbp and cdc hparams cannot be both absent')
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

    params0 = Params(mf, dbp)
    inpbuf0 = InpBuf(ybuf, xbuf, fobuf)
    afstate0 = AFStat(mimostat)

    @jit
    def updinpbuf(inp, inpbuf: InpBuf):
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

    def tddbp(y, d, n, backend=None):
        dbpfn = lambda a, b, c: xcomm.dbp_timedomain(a, b, c, mode='valid', homosteps=True, scansteps=True)
        return jit(dbpfn, backend=backend)(y, d, n)

    @jit
    def iterate(inp, params: Params, inpbuf: InpBuf, afstate: AFStat):
        inp, inpbuf = updinpbuf(inp, inpbuf)

        y, x, fomul = inp
        mf, dbp = params
        mimostat, = afstate

        if dbphparams is not None:
            z = tddbp(y, dbp.d, dbp.n)
        else:
            z = eq.cdcomp(y, cdchparams.sr, cdchparams.cd, mode='valid')
        z *= fomul
        z = matchedfilter(z, mf, mode='valid')
        zf = xop.frame(z, lmshparams.taps, sps)
        mimostat, (mimovals, _) = af.iterate(lms_update, mimostat, zf, x)
        z = lms_map(mimovals, zf)

        afstate = AFStat(mimostat,)
        afvals = AFVals(mimovals,)
        return z, inpbuf, afstate, afvals

    @jit
    def model(inp, params: Params, inpbuf: InpBuf, afvals: AFVals):
        inp, inpbuf = updinpbuf(inp, inpbuf)

        y, _, fomul = inp
        mf, dbp = params
        mimovals, = afvals

        if dbphparams is not None:
            z = tddbp(y, dbp.d, dbp.n)
        else:
            z = eq.cdcomp(y, cdchparams.sr, cdchparams.cd, mode='valid')
        z *= fomul
        z = matchedfilter(z, mf, mode='valid')
        zf = xop.frame(z, lmshparams.taps, sps)
        z = lms_map(mimovals, zf)
        return z, inpbuf

    def lossfn(v, z, inp, params: Params, inpbuf: InpBuf, afvals: AFVals):
        zhat, _ = model(inp, params._replace(**v._asdict()), inpbuf, afvals)
        return jnp.mean(jnp.abs(zhat - z)**2)

    @jit
    def value_and_grad(v, z, inp, params: Params, inpbuf: InpBuf, afvals: AFVals):
        loss, grads = jax.value_and_grad(lossfn)(v, z, inp, params, inpbuf, afvals)
        return loss, grads

    def eziter(inp, params: Params=params0, zerodelayroll=True, backend='cpu'):
        z = jit(iterate, backend=backend)(inp, params, inpbuf0, afstate0)[0]
        return xop.delay(z, -dspdelay) if zerodelayroll else z

    return DSPModel(params0, inpbuf0, afstate0, dspdelay, eziter, iterate, model, value_and_grad)



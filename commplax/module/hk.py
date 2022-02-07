import numpy as np
import haiku as hk
from jax import numpy as jnp
from commplax import xop, comm, xcomm, util, adaptive_filter as af, equalizer as eq
from functools import partial


def transform_af(f, **partial_kwargs):
    return hk.without_apply_rng(hk.transform_with_state(partial(f, **partial_kwargs)))


class CMA(hk.Module):

  def __init__(self, lr=2**-14, const="16QAM", name=None):
    super().__init__(name=name)
    self.lr = lr
    self.const = jnp.asarray(comm.const(const, norm=True) if isinstance(const, str) else const)
    self.R2 = jnp.asarray(np.mean(abs(self.const)**4) / np.mean(abs(self.const)**2))
    self._init, self._update, self._apply = af.cma(R2=self.R2, lr=self.lr)

  def __call__(self, framed_signal):
    taps, dims = framed_signal.shape[1], framed_signal.shape[2]
    # get initial state
    w = hk.get_state("w", shape=(dims, dims, taps), dtype=framed_signal.dtype, init=self._init)
    i = hk.get_state("i", shape=tuple(), dtype=jnp.int32, init=jnp.zeros)
    # run CMA
    i, (w, (ws, ls)) = af.iterate(self._update, i, w, framed_signal)
    # apply weights
    signal = self._apply(ws, framed_signal)
    # update states
    hk.set_state("w", w)
    hk.set_state("i", i + 1)

    self.aux_out = {"weights": ws, "losses": ls}

    return signal


class MUCMA(hk.Module):

  def __init__(self, lr=2**-13, delta=6, beta=0.999, const="16QAM", name=None):
    super().__init__(name=name)
    self.lr = lr
    self.const = jnp.asarray(comm.const(const, norm=True) if isinstance(const, str) else const)
    self.R2 = jnp.asarray(np.mean(abs(self.const)**4) / np.mean(abs(self.const)**2))
    self.delta = delta
    self.beta = beta
    self._init, self._update, self._apply = af.mucma(R2=self.R2, lr=self.lr)

  def __call__(self, framed_signal):
    taps, dims = framed_signal.shape[1], framed_signal.shape[2]
    # get initial state
    w = hk.get_state("w", shape=(dims, dims, taps), dtype=framed_signal.dtype, init=lambda *a: self._init(*a)[0])
    z = hk.get_state("z", shape=(self.delta, dims), dtype=framed_signal.dtype, init=jnp.zeros)
    r = hk.get_state("r", shape=(dims, dims, self.delta), dtype=framed_signal.dtype, init=jnp.zeros)
    bp = hk.get_state("bp", shape=(), dtype=jnp.float32, init=lambda *a: jnp.array(self.beta))
    i = hk.get_state("i", shape=tuple(), dtype=jnp.int32, init=jnp.zeros)
    # run CMA
    state = (w, z, r, bp)
    i, (state, (ws, ls)) = af.iterate(self._update, i, state, framed_signal)
    w, z, r, bp = state
    # apply weights
    signal = self._apply(ws, framed_signal)
    # update states
    hk.set_state("w", w)
    hk.set_state("z", z)
    hk.set_state("r", r)
    hk.set_state("bp", bp)
    hk.set_state("i", i + 1)

    self.aux_out = {"weights": ws, "losses": ls}

    return signal


class DDLMS(hk.Module):

    def __init__(self,
                 train=False,
                 lr_w=1/2**4,
                 lr_f=1/2**7,
                 lr_s=0.,
                 lr_b=1/2**11,
                 const="16QAM",
                 name=None):
        super().__init__(name=name)
        self.const = jnp.asarray(comm.const(const, norm=True) if isinstance(const, str) else const)
        self._init, self._update, self._apply = af.ddlms(lr_w=lr_w, lr_f=lr_f, lr_s=lr_s, lr_b=lr_b,
                                                         train=train, const=self.const)

    def __call__(self, framed_signal, truth=None):
        taps, dims = framed_signal.shape[1], framed_signal.shape[2]
        # get initial state
        w = hk.get_state("w", shape=(dims, dims, taps), dtype=framed_signal.dtype, init=jnp.zeros)
        f = hk.get_state("f", shape=(dims,), dtype=framed_signal.dtype, init=jnp.ones)
        s = hk.get_state("s", shape=(dims,), dtype=framed_signal.dtype, init=jnp.ones)
        b = hk.get_state("b", shape=(dims,), dtype=framed_signal.dtype, init=jnp.zeros)
        fshat = hk.get_state("fshat", shape=(dims,), dtype=framed_signal.dtype, init=jnp.ones)
        i = hk.get_state("i", shape=tuple(), dtype=jnp.int32, init=jnp.zeros)
        # run CMA
        state = (w, f, s, b, fshat)
        i, (state, (ws, ls)) = af.iterate(self._update, i, state, framed_signal, truth)
        w, f, s, b, fshat = state
        # apply weights
        signal = self._apply(ws, framed_signal)
        # update states
        hk.set_state("w", w)
        hk.set_state("f", f)
        hk.set_state("s", s)
        hk.set_state("b", b)
        hk.set_state("fshat", fshat)
        hk.set_state("i", i + 1)

        self.aux_out = {"weights": ws, "losses": ls}

        return signal


class FrameEKFCPR(hk.Module):
    
    def __init__(self, const="16QAM", w0=0., name=None):
        super().__init__(name=name)
        self.const = jnp.asarray(comm.const(const, norm=True) if isinstance(const, str) else const)
        self.w0 = w0
        self.dims = None

    def __call__(self, signal):
        block_size, dims = signal.shape[1], signal.shape[2]
        if not self.dims == dims: 
            self._init, self._update, self._apply = af.array(af.frame_cpr_kf, dims)(alpha=0.98,
                                                                                    R=jnp.array([[1e-2, 0],
                                                                                                 [0, 1e-4]]),
                                                                                    const=self.const)
            self.dims = dims
        # get initial state
        z0 = hk.get_state("z", shape=(), dtype=signal.dtype, init=lambda *_: self._init(self.w0)[0])
        P0 = hk.get_state("P", shape=(), dtype=signal.dtype, init=lambda *_: self._init(0)[1])
        Q0 = hk.get_state("Q", shape=(), dtype=signal.dtype, init=lambda *_: self._init(0)[2])
        i = hk.get_state("i", shape=tuple(), dtype=jnp.int32, init=jnp.zeros)
        # run EKF-FOE
        state = (z0, P0, Q0)
        i, (state, (ws, phis)) = af.iterate(self._update, i, state, signal)
        z, P, Q = state
        # apply weights
        signal = self._apply(phis, signal).reshape((-1, dims))
        # update states
        hk.set_state("z0", z0)
        hk.set_state("P0", P0)
        hk.set_state("Q", Q)
        hk.set_state("i", i + 1)

        self.aux_out = {"ws": ws.reshape((-1, dims)), "phis": phis.reshape((-1, dims))}

        return signal


class CPANEEKFCPR(hk.Module):
    
    def __init__(self, const="16QAM", name=None):
        super().__init__(name=name)
        self.const = jnp.asarray(comm.const(const, norm=True) if isinstance(const, str) else const)
        self.dims = None

    def __call__(self, signal):
        dims = signal.shape[1]
        if not self.dims == dims: 
            self._init, self._update, self._apply = af.array(af.cpane_ekf, dims)(beta=0.6, const=self.const)
            self.dims = dims
        # get initial state
        Psi_c = hk.get_state("Psi_c", shape=(), dtype=signal.dtype, init=lambda *_: self._init()[0])
        P_c   = hk.get_state("P_c",   shape=(), dtype=signal.dtype, init=lambda *_: self._init()[1])
        Psi_a = hk.get_state("Psi_a", shape=(), dtype=signal.dtype, init=lambda *_: self._init()[2])
        Q     = hk.get_state("Q",     shape=(), dtype=signal.dtype, init=lambda *_: self._init()[3])
        R     = hk.get_state("R",     shape=(), dtype=signal.dtype, init=lambda *_: self._init()[4])
        i = hk.get_state("i", shape=tuple(), dtype=jnp.int32, init=jnp.zeros)
        # run EKF-FOE
        state = (Psi_c, P_c, Psi_a, Q, R)
        i, (state, (phis, _)) = af.iterate(self._update, i, state, signal)
        Psi_c, P_c, Psi_a, Q, R = state
        # apply weights
        signal = self._apply(phis, signal).reshape((-1, dims))
        # update states
        hk.set_state("Psi_c", Psi_c)
        hk.set_state("P_c", P_c)
        hk.set_state("Psi_a", Psi_a)
        hk.set_state("Q", Q)
        hk.set_state("R", R)
        hk.set_state("i", i + 1)

        self.aux_out = {"phi": phis}

        return signal

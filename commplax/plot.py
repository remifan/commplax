import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


def glance(x):
    x = np.asarray(x)

    if x.ndim > 1:
        nch = x.shape[-1]
    else:
        nch = 1
        x = x[:,None]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, nch)

    asp = lambda gs: fig.add_subplot(gs)

    for ch in range(nch):
        pwelch(x[:,ch], ax=asp(gs[0, ch]))
        waveform(x[:,ch], axes=[asp(gs[1, ch]), asp(gs[2, ch])])


def scatter(signal, title="Constellation", ax=None, density=False):
    if np.iscomplex(signal).any():
        I = np.real(signal)
        Q = np.imag(signal)
    else:
        I = signal[0]
        Q = signal[1]

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(3)
        fig.set_figheight(3)
        fig.set_dpi(100)
    #plt.figure(num=None, figsize=(8, 6), dpi=100)
    if density:
        ax = sns.kdeplot(x=I, y=Q, fill=True, ax=ax)
    else:
        ax.scatter(I, Q, s=1)
    ax.axis('equal')
    ax.set_title(title)


def pwelch(x, ax=None):
    f, w = signal.welch(x, return_onesided=False)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(f, abs(w))


def waveform(x, axes=None):
    x = np.asarray(x)

    if axes is None:
        fig, axes = plt.subplots(2,1)
    axes[0].plot(x.real)
    axes[0].plot([x.real.mean().tolist(),] * len(x))
    axes[1].plot(x.imag)
    axes[1].plot([x.imag.mean().tolist(),] * len(x))


def desc_filter(w, H, ax=None, color=None, legend=None, phase=True):
    if ax is None:
        fig = plt.figure()
        ax1 = plt.gca()
    else:
        ax1 = ax

    ax2 = None

    if H.ndim == 1:
        H = H[None,:]

    lgd   = tuple(map(str, np.arange(len(H)))) if legend is None else legend
    lgd   = [lgd] if not isinstance(lgd, list) and not isinstance(lgd, tuple) else lgd
    color = [color] if not isinstance(color, list) and not isinstance(color, tuple) else color

    for H_,i in zip(H, range(len(H))):
        ax1.plot(w, abs(H_))
        ax1.set_ylabel('amp.')

        if phase:
            if ax2 is None:
                ax2 = ax1.twinx()
            ax2.plot(w, np.unwrap(np.angle(H_)), alpha=0.4)
            ax2.set_ylabel('phs.')
    ax1.legend(lgd, loc='best')


def lpvssnr(LP, S, label=None, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if S.ndim == 3:
        S = S.mean(axis=-1)
    ax.errorbar(LP, S[:, 0], yerr=S[:, 2:].T, label=label, fmt='-o', capsize=3)
    ax.fill_between(LP, S[:, 0] - S[:, 1], S[:, 0] + S[:, 1], alpha=0.2)
    ax.legend()
    ax.set_xlabel('LP (dBm)')
    ax.set_ylabel('SNR (dB)')



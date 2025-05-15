# Copyright 2021 The Commplax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.cluster.vq import kmeans


def glance(x, show_spectrum=True, show_waveform=True):
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
        if show_spectrum:
            pwelch(x[:,ch], ax=asp(gs[0, ch]))
        if show_waveform:
            waveform(x[:,ch], axes=[asp(gs[1, ch]), asp(gs[2, ch])])


def _const_optim(x, c):
    u = np.stack([x.real, x.imag], axis=-1)
    v = np.stack([c.real, c.imag], axis=-1)
    codebook, distortion = kmeans(u, v, iter=50, thresh=1e-08)
    const = codebook[:, 0] + 1j*codebook[:, 1]
    return const

def scatter(signal, kde=False, kdeopts={'fill': True, 'cmap':'Reds'}, title=None, const=None, dpi=100):
    signal = np.atleast_1d(signal)
    # if not np.iscomplex(signal).all():
    #     raise ValueError(f'expect complex input, got {signal.dtype} instead')
      
    if signal.ndim == 1:
      signal = signal[:, None]
      
    dims = signal.shape[1]
    
    fig, axes = plt.subplots(1, dims, dpi=dpi, sharex=True, sharey=True)
    
    if dims == 1:
        axes = [axes]
    
    for i in range(dims):
        ax = axes[i]
        x = signal[:, i]
      
        I = np.real(x)
        Q = np.imag(x)

        if kde:
            ax = sns.kdeplot(x=I, y=Q, ax=ax, **kdeopts)
        else:
            ax.scatter(I, Q, s=1)
            if const is not None:
                ax.scatter(const.real, const.imag, marker='x', color='red')
                const_kmeans = _const_optim(x, const)
                ax.scatter(const_kmeans.real, const_kmeans.imag, marker='+', color='black')
                

        ax.set_aspect('equal')

        if dims > 1:
            ax.set_title(f'dim{i}') if title is None else ax.set_title(title[i])


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


def desc_filter(w, H, ax=None, colors=None, legend=None, Hermitian=False, phase=True,
                legend_fontsize=9):
    if ax is None:
        fig = plt.figure()
        ax1 = plt.gca()
    else:
        ax1 = ax

    ax2 = None

    if H.ndim == 1:
        H = H[None,:]

    if Hermitian:
        phase = False

    lgd   = tuple(map(str, np.arange(len(H)))) if legend is None else legend
    lgd   = [lgd] if not isinstance(lgd, list) and not isinstance(lgd, tuple) else lgd
    colors = [colors] * len(H) if colors is None else colors

    for H_,i in zip(H, range(len(H))):
        if Hermitian:
            ax1.plot(w, abs(H_) * np.sign(np.exp(1j * np.unwrap(np.angle(H_))).real), color=colors[i])
        else:
            ax1.plot(w, abs(H_), color=colors[i])
        ax1.set_ylabel('amplitude (a.u.)')
        ax1.set_xlabel('freq. (Hz)')

        if phase:
            if ax2 is None:
                ax2 = ax1.twinx()
            ax2.plot(w, np.unwrap(np.angle(H_)), '--', color=colors[i], alpha=0.4)
            ax2.set_ylabel('phase (rad)')
    if len(H) > 1:
        ax1.legend(lgd, loc='best', fontsize=legend_fontsize)


def lpvssnr(LP, S, label=None, ax=None, show_std=True, show_ex=True):
    S = np.stack(S)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if S.ndim == 3:
        S = S.mean(axis=-1)
    if show_ex:
        ax.errorbar(LP, S[:, 0], yerr=S[:, 2:].T, label=label, fmt='-o', capsize=3)
    else:
        ax.plot(LP, S[:, 0], '-o', label=label)
    if show_std:
        ax.fill_between(LP, S[:, 0] - S[:, 1], S[:, 0] + S[:, 1], alpha=0.2)
    ax.legend()
    ax.set_xlabel('Launched Power (dBm)')
    ax.set_ylabel('SNR (dB)')


def filter_response(b):
    w, h = signal.freqz(b)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')

    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()
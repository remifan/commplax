import numpy as np
from functools import partial

newaxis = np.newaxis
fft = partial(np.fft.fft, axis=0)
ifft = partial(np.fft.ifft, axis=0)
fftfreq = np.fft.fftfreq
dBm2W = lambda x: 10**(x / 10) * 1e-3
W2dBm = lambda x: 10 * np.log10(x / 1e-3)
dB2lin = lambda x: 10**(x / 10)
lin2dB = lambda x: 10 * np.log10(x)
vectorize_element_wise_op2 = lambda op: np.vectorize(lambda a, b: op(a, b), signature='(n),(n)->(n)')
vmul = lambda a, b: vectorize_element_wise_op2(np.multiply)(a.T, b.T).T
vadd = lambda a, b: vectorize_element_wise_op2(np.add)(a.T, b.T).T


def SSF(
    E_t,
    f_s,
    L,
    a=.2E-3,
    D_λ=16E-6,
    λ_ref=1552.52e-9,
    n_2=2.6E-20,
    A_eff=80E-12,
    hz_max=1e3,
    hz_min=.0,
    Φ_max=0.05):

    Nfft = E_t.shape[0]

    c = 299792458   # speed of light (vacuum)
    α = a / (10. / np.log(10.))
    γ = 2 * np.pi * n_2 / λ_ref / A_eff
    β_2 = -(D_λ * λ_ref**2)/(2 * np.pi * c)
    ω = 2 * np.pi * f_s * fftfreq(Nfft)
    
    Epow = lambda E: np.abs(E)**2

    # calculate bounded step size given signal power
    calc_hz = lambda E_t: np.clip(Φ_max / (γ * Epow(E_t).max()),
                                  hz_min,
                                  hz_max)

    # linear (D) and nonlinear (N) operators
    lin_op = lambda E_f, hz: E_f * \
        np.exp(-(α / 2) * (hz / 2) + 1j * (β_2 / 2) * (ω**2) * (hz / 2))

    nli_op = lambda E_t, hz: E_t * np.exp(1j * γ * Epow(E_t) * hz)

    # symmetric ssfm: ...(D-N-D)-(D-N-D)-..., where signal starts and ends at
    # frequency domain for each step
    hz_new = calc_hz(E_t)
    E_f = fft(E_t)

    while L > 0:
        # update split step size
        hz = hz_new if L >= hz_new else L

        # 1st linear step (frequency domain)
        E_f = lin_op(E_f, hz)

        # Nonlinear step (time domain)
        E_t = ifft(E_f)
        E_t = nli_op(E_t, hz)

        # new step size is supposed to be calculated at beginning of each iteration,
        # which needs extra fft/ifft. Calculating step size at middle step should be
        # safe since signal power is unlikely to vary much after single nonlinear step.
        hz_new = calc_hz(E_t)
        
        # 2nd linear step (frequency domain)
        E_f = fft(E_t)
        E_f = lin_op(E_f, hz)

        L = L - hz

    E_t = ifft(E_f)

    return E_t


def manakovSSF(
    E_t,
    f_s,
    L,
    a=.2E-3,
    D_λ=16E-6,
    λ_ref=1552.52e-9,
    n_2=2.6E-20,
    A_eff=80E-12,
    hz_max=1e3,
    hz_min=.0,
    Φ_max=0.05):

    Nfft = E_t.shape[0]

    c = 299792458   # speed of light (vacuum)
    α = a / (10. / np.log(10.))
    γ = 2 * np.pi * n_2 / λ_ref / A_eff
    β_2 = -(D_λ * λ_ref**2)/(2 * np.pi * c)
    ω = 2 * np.pi * f_s * fftfreq(Nfft)
    
    Epow = lambda E: (np.abs(E)**2).sum(axis=-1)

    # calculate bounded step size given signal power
    calc_hz = lambda E_t: np.clip(Φ_max / (γ * 8/9 * Epow(E_t).max()),
                                  hz_min,
                                  hz_max)

    # linear (D) and nonlinear (N) operators
    lin_op = lambda E_f, hz: E_f * \
        np.exp(-(α / 2) * (hz / 2) + 1j * (β_2 / 2) * (ω[:, newaxis]**2) * (hz / 2))

    nli_op = lambda E_t, hz: E_t * \
        np.exp(1j * γ * 8/9 * Epow(E_t) * hz)[:, newaxis]

    # symmetric ssfm: ...(D-N-D)-(D-N-D)-..., where signal starts and ends at
    # frequency domain for each step
    hz_new = calc_hz(E_t)
    E_f = fft(E_t)

    while L > 0:
        # update split step size
        hz = hz_new if L >= hz_new else L

        # 1st linear step (frequency domain)
        E_f = lin_op(E_f, hz)

        # Nonlinear step (time domain)
        E_t = ifft(E_f)
        E_t = nli_op(E_t, hz)

        # new step size is supposed to be calculated at beginning of each iteration,
        # which needs extra fft/ifft. Calculating step size at middle step should be
        # safe since signal power is unlikely to vary much after single nonlinear step.
        hz_new = calc_hz(E_t)
        
        # 2nd linear step (frequency domain)
        E_f = fft(E_t)
        E_f = lin_op(E_f, hz)

        L = L - hz

    E_t = ifft(E_f)

    return E_t


def EDFA(E_t, fs, gain, nf=5, fc=193.1e12):
    h          = 6.626e-34 # Planck constant
    nf_lin     = dB2lin(nf)
    gain_lin   = dB2lin(gain)
    nsp        = (gain_lin * nf_lin - 1) / (2 * (gain_lin - 1))
    s_ase      = (gain_lin - 1) * nsp * h * fc
    p_noise    = s_ase * fs
    mean_noise = 0
    noise      = np.random.normal(mean_noise, np.sqrt(p_noise), E_t.shape) +\
        1j * np.random.normal(mean_noise, np.sqrt(p_noise), E_t.shape)

    return E_t * np.sqrt(gain_lin) + noise


def laserPhaseNoise(E_t, df, fs):
    var = 2*np.pi*df/fs
    f = np.random.normal(scale=np.sqrt(var), size=E_t.shape)
    pn =  np.cumsum(f, axis=0)
    E_t = E_t * np.exp(1.j * pn)
    return E_t


def laserFreqOffset(E_t, fo, fs):
    ph = np.exp(2j * np.pi * np.arange(E_t.shape[0]) * fo / fs)
    return E_t * ph[:, newaxis] if E_t.ndim > 1 else E_t * ph



# Copyright 2026 The Commplax Authors.
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

"""Global timing analysis and acquisition methods.

This module provides FFT-based global methods for symbol timing acquisition
and analysis. These are primarily for debugging and initialization, not
real-time DSP. For runtime symbol timing recovery, see `commplax.sym_timing`.

Functions:
    find_clock: FFT-based clock frequency detection and resampling
    clock_spectrum: Compute clock spectrum for visualization
    analyze_clock: Clock recovery with diagnostic output

Example:
    >>> from commplax.analyzers.timing import find_clock, clock_spectrum
    >>> import numpy as np
    >>>
    >>> # Detect clock and resample to 1 sps
    >>> result = find_clock(signal_2sps, sps=2.0)
    >>> signal_1sps = result['signal']
    >>> print(f"Actual SPS: {result['sps_actual']:.6f}")
    >>>
    >>> # Visualize clock spectrum
    >>> freqs, spectrum = clock_spectrum(signal_2sps)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import PchipInterpolator


def _fitline(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Linear regression: y = slope * x + intercept.

    Args:
        x: Independent variable array
        y: Dependent variable array

    Returns:
        Tuple of (slope, intercept)
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    denom = n * sum_x2 - sum_x ** 2
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _interp_at_t(signal: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate signal at specified time points using PCHIP.

    Args:
        signal: Input signal array, shape (N,) or (N, D)
        t: Time points to interpolate at (fractional sample indices)

    Returns:
        Interpolated signal values at times t
    """
    n = len(signal)
    x = np.arange(n)

    if signal.ndim == 1:
        if np.iscomplexobj(signal):
            interp_re = PchipInterpolator(x, signal.real)
            interp_im = PchipInterpolator(x, signal.imag)
            return interp_re(t) + 1j * interp_im(t)
        else:
            interp = PchipInterpolator(x, signal)
            return interp(t)
    else:
        # Multi-dimensional signal (N, D)
        result = np.zeros((len(t), signal.shape[1]), dtype=signal.dtype)
        for d in range(signal.shape[1]):
            col = signal[:, d]
            if np.iscomplexobj(col):
                interp_re = PchipInterpolator(x, col.real)
                interp_im = PchipInterpolator(x, col.imag)
                result[:, d] = interp_re(t) + 1j * interp_im(t)
            else:
                interp = PchipInterpolator(x, col)
                result[:, d] = interp(t)
        return result


def clock_spectrum(
    signal: np.ndarray,
    nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute clock spectrum (power spectrum of |signal|^2).

    The clock tone appears at the symbol rate in the power spectrum due to
    the signal's cyclostationary property.

    Args:
        signal: Input signal, shape (N,) or (N, D) for D polarizations
        nfft: FFT size (default: signal length)

    Returns:
        freqs: Normalized frequency array (0 to 1, where 1 = sample rate)
        spectrum: Magnitude spectrum (linear scale)
    """
    signal = np.atleast_2d(signal.T).T  # ensure (N, D) shape
    n = signal.shape[0]

    if nfft is None:
        nfft = n

    # Compute power (squared magnitude) - clock tone appears here
    power = np.sum(np.abs(signal) ** 2, axis=1)

    # FFT of power signal
    spectrum = np.fft.fft(power, n=nfft)
    freqs = np.fft.fftfreq(nfft)

    return freqs, np.abs(spectrum)


def find_clock(
    signal: np.ndarray,
    sps: float,
    search_range: float = 0.01,
    nfft: Optional[int] = None,
    resample: bool = True,
) -> dict:
    """Find clock frequency and optionally resample to symbol rate.

    Uses power spectrum peak detection to find the clock tone at the symbol
    rate, estimates phase, and resamples the signal to align with symbol
    timing.

    Algorithm:
        1. Compute power spectrum of |signal|^2 (clock tone at symbol rate)
        2. Find peak near expected clock frequency (1/sps)
        3. Refine frequency estimate using parabolic interpolation
        4. Estimate phase from complex spectrum
        5. Resample signal to align with detected symbol timing

    Args:
        signal: Input signal, shape (N,) or (N, D) for D polarizations
        sps: Nominal samples per symbol
        search_range: Fractional range to search around expected clock freq
            (default: 0.01 = +/- 1%)
        nfft: FFT size for frequency estimation (default: signal length)
        resample: If True, resample signal to 1 sps (default: True)

    Returns:
        Dictionary with:
            'signal': Resampled signal at 1 sps (if resample=True, else None)
            'f_clock': Detected clock frequency (normalized to sample rate)
            'phase': Clock phase estimate in radians
            'sps_actual': Actual samples per symbol (1/f_clock)
            'sps_error_ppm': SPS error in parts per million
            't_resample': Resampling time points (if resample=True)

    Raises:
        ValueError: If no frequencies found in search range

    Example:
        >>> signal_2sps = ...  # 2 samples per symbol
        >>> result = find_clock(signal_2sps, sps=2.0)
        >>> signal_1sps = result['signal']
        >>> print(f"SPS error: {result['sps_error_ppm']:.2f} ppm")
    """
    signal = np.atleast_2d(signal.T).T  # ensure (N, D) shape
    n, d = signal.shape

    if nfft is None:
        nfft = n

    # Expected clock frequency (normalized to sample rate)
    f_expected = 1.0 / sps

    # Get clock spectrum
    freqs, spectrum_mag = clock_spectrum(signal, nfft)
    spectrum = np.fft.fft(np.sum(np.abs(signal) ** 2, axis=1), n=nfft)

    # Search range around expected clock frequency
    f_low = f_expected * (1 - search_range)
    f_high = f_expected * (1 + search_range)

    # Find indices in search range (positive frequencies only)
    mask = (freqs >= f_low) & (freqs <= f_high)
    search_indices = np.where(mask)[0]

    if len(search_indices) == 0:
        raise ValueError(
            f"No frequencies found in search range [{f_low:.6f}, {f_high:.6f}]. "
            f"Expected clock at {f_expected:.6f} (sps={sps})"
        )

    # Find peak in search range
    peak_idx = search_indices[np.argmax(spectrum_mag[search_indices])]
    f_clock = freqs[peak_idx]

    # Refine frequency estimate using parabolic interpolation
    if 0 < peak_idx < nfft - 1:
        alpha = spectrum_mag[peak_idx - 1]
        beta = spectrum_mag[peak_idx]
        gamma = spectrum_mag[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            delta = 0.5 * (alpha - gamma) / denom
            f_clock = freqs[peak_idx] + delta * (freqs[1] - freqs[0])

    # Estimate phase from complex spectrum
    clock_phase = np.angle(spectrum[peak_idx])

    # Actual samples per symbol
    sps_actual = 1.0 / f_clock
    sps_error_ppm = (sps_actual - sps) / sps * 1e6

    result = {
        'f_clock': f_clock,
        'phase': clock_phase,
        'sps_actual': sps_actual,
        'sps_error_ppm': sps_error_ppm,
        'signal': None,
        't_resample': None,
    }

    if resample:
        # Generate resampling time points at symbol centers
        # Start phase determines first symbol position
        t0 = -clock_phase / (2 * np.pi * f_clock)
        if t0 < 0:
            t0 += sps_actual
        n_symbols = int((n - t0) / sps_actual)
        t_resample = t0 + np.arange(n_symbols) * sps_actual

        # Clip to valid range
        t_resample = t_resample[(t_resample >= 0) & (t_resample < n - 1)]

        # Resample signal
        resampled = _interp_at_t(signal, t_resample)

        # Squeeze if input was 1D
        if d == 1:
            resampled = resampled.squeeze(axis=1)

        result['signal'] = resampled
        result['t_resample'] = t_resample

    return result


def analyze_clock(
    signal: np.ndarray,
    sps: float,
    label: str = "Clock Recovery",
    verbose: bool = True,
) -> dict:
    """Analyze clock recovery and optionally print diagnostics.

    Convenience wrapper around find_clock() that provides formatted output.

    Args:
        signal: Input signal
        sps: Nominal samples per symbol
        label: Label for diagnostic output
        verbose: If True, print diagnostics (default: True)

    Returns:
        Clock recovery results (same as find_clock)
    """
    result = find_clock(signal, sps)

    if verbose:
        print(f"{label}:")
        print(f"  Expected SPS: {sps:.6f}")
        print(f"  Actual SPS:   {result['sps_actual']:.6f}")
        print(f"  Clock freq:   {result['f_clock']:.6f} (normalized)")
        print(f"  Clock phase:  {result['phase']:.4f} rad")
        print(f"  SPS error:    {result['sps_error_ppm']:.2f} ppm")

    return result

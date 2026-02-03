"""
Signal visualization utilities.

This module provides plotting functions for visualizing communication signals
in various representations (time domain, frequency domain, constellation, etc.).
"""

import numpy as np
from typing import Optional, List, Tuple

# Plotly imports are deferred to avoid hard dependency
_plotly_available = None


def _check_plotly():
    """Check if plotly is available, raise ImportError if not."""
    global _plotly_available
    if _plotly_available is None:
        try:
            import plotly.graph_objects  # noqa: F401
            _plotly_available = True
        except ImportError:
            _plotly_available = False
    if not _plotly_available:
        raise ImportError(
            "plotly is required for this function. "
            "Install it with: pip install plotly"
        )


def plot_constellation_3d(
    symbols: np.ndarray,
    time: Optional[np.ndarray] = None,
    highlight_intervals: Optional[List[Tuple[int, int]]] = None,
    base_color: str = 'rgba(150, 150, 150, 0.4)',
    highlight_color: str = 'rgba(220, 20, 60, 0.9)',
    marker_size: float = 1,
    width: Optional[int] = None,
    height: int = 450,
    title: Optional[str] = None,
    aspect_ratio: Tuple[float, float, float] = (2.5, 1, 1),
    camera_eye: Tuple[float, float, float] = (-1.8, -1.2, 0.5),
    subplot_titles: Optional[List[str]] = None,
    show_density: bool = False,
    density_bins: int = 64,
    density_colorscale: str = 'Blues',
):
    """
    Visualize complex symbols in 3D with time as x-axis, I/Q as y/z-axes.

    Parameters
    ----------
    symbols : np.ndarray
        Complex symbols array. Shape can be:
        - (N,): single channel, produces one 3D plot
        - (N, K): K channels, produces K vertically stacked subplots
    time : np.ndarray, optional
        Time indices for each symbol. If None, uses np.arange(N).
    highlight_intervals : list of tuple, optional
        List of (start, end) index intervals to highlight. Supports negative
        indices (e.g., (-100, -1) for last 100 symbols).
        If None (default), auto-generates 5 evenly spaced intervals.
        Set to [] to disable highlighting.
    base_color : str, default 'rgba(150, 150, 150, 0.4)'
        Color for non-highlighted symbols (RGBA string).
    highlight_color : str, default 'rgba(220, 20, 60, 0.9)'
        Color for highlighted symbols (RGBA string).
    marker_size : float, default 1
        Size of the dot markers.
    width : int, optional
        Figure width in pixels. If None (default), auto-sizes to container.
    height : int, default 450
        Figure height in pixels (per channel).
    title : str, optional
        Plot title. If None, no title is shown.
    aspect_ratio : tuple, default (2.5, 1, 1)
        Aspect ratio for the 3D box (x, y, z). Increase x to stretch time axis.
    camera_eye : tuple, default (-1.8, -1.2, 0.5)
        Camera position (x, y, z). Controls the viewing angle.
    subplot_titles : list of str, optional
        Titles for each subplot. If None, uses "Channel 0", "Channel 1", etc.
    show_density : bool, default False
        If True, show a 2D density heatmap at the base of the plot using
        the highlighted symbols.
    density_bins : int, default 64
        Number of bins for density histogram (per axis).
    density_colorscale : str, default 'Blues'
        Colorscale for density coloring (e.g., 'Blues', 'Purples', 'Viridis').

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> import numpy as np
    >>> from commplax.analyzers.visualize import plot_constellation_3d

    Single channel:

    >>> symbols = np.random.randn(1000) + 1j * np.random.randn(1000)
    >>> fig = plot_constellation_3d(symbols)
    >>> fig.show()

    Multi-channel (e.g., dual polarization):

    >>> symbols_2ch = np.stack([symbols, symbols * np.exp(1j * 0.1)], axis=1)
    >>> fig = plot_constellation_3d(symbols_2ch, subplot_titles=['X-pol', 'Y-pol'])
    >>> fig.show()

    Custom highlight intervals:

    >>> fig = plot_constellation_3d(symbols, highlight_intervals=[(0, 100), (-100, -1)])
    >>> fig.show()
    """
    _check_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    symbols = np.asarray(symbols)

    # Handle dimensions: ensure shape is (N, K)
    if symbols.ndim == 1:
        symbols = symbols[:, np.newaxis]  # (N,) -> (N, 1)

    n_symbols, n_channels = symbols.shape

    if time is None:
        time = np.arange(n_symbols)

    if len(time) != n_symbols:
        raise ValueError("time must have the same length as symbols")

    # Auto-generate 5 evenly spaced intervals if not provided
    if highlight_intervals is None:
        # Window size: 3% of symbols, clamped between 60 and 600
        window_size = max(60, min(600, n_symbols // 33))
        half_window = window_size // 2
        # 5 center positions at 0%, 25%, 50%, 75%, 100%
        centers = [0, n_symbols // 4, n_symbols // 2, 3 * n_symbols // 4, n_symbols - 1]
        highlight_intervals = []
        for center in centers:
            start = max(0, center - half_window)
            end = min(n_symbols - 1, center + half_window)
            highlight_intervals.append((start, end))

    # Build highlight mask
    highlight_mask = np.zeros(n_symbols, dtype=bool)
    if highlight_intervals:
        for start, end in highlight_intervals:
            # Handle negative indices
            if start < 0:
                start = n_symbols + start
            if end < 0:
                end = n_symbols + end
            # Clamp to valid range
            start = max(0, start)
            end = min(n_symbols - 1, end)
            highlight_mask[start:end + 1] = True

    # Default subplot titles
    if subplot_titles is None:
        subplot_titles = [f'Channel {i}' for i in range(n_channels)]

    # Create figure with subplots (vertical layout)
    if n_channels == 1:
        fig = go.Figure()
    else:
        specs = [[{'type': 'scatter3d'}] for _ in range(n_channels)]
        fig = make_subplots(
            rows=n_channels, cols=1,
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
        )

    base_mask = ~highlight_mask

    for ch in range(n_channels):
        I = symbols[:, ch].real
        Q = symbols[:, ch].imag

        # Scene name for this subplot
        scene_name = 'scene' if ch == 0 else f'scene{ch + 1}'

        # Plot non-highlighted points (base layer)
        if np.any(base_mask):
            trace = go.Scatter3d(
                x=time[base_mask],
                y=I[base_mask],
                z=Q[base_mask],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=base_color,
                ),
                name='Symbols' if ch == 0 else None,
                showlegend=(ch == 0),
            )
            if n_channels == 1:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=ch + 1, col=1)

        # Plot highlighted points (with optional density coloring)
        if np.any(highlight_mask):
            I_hl = I[highlight_mask]
            Q_hl = Q[highlight_mask]
            t_hl = time[highlight_mask]

            if show_density:
                # Compute local density using 2D histogram lookup
                iq_range = max(np.abs(I_hl).max(), np.abs(Q_hl).max()) * 1.1
                hist, i_edges, q_edges = np.histogram2d(
                    I_hl, Q_hl, bins=density_bins,
                    range=[[-iq_range, iq_range], [-iq_range, iq_range]]
                )
                # Find bin indices for each point
                i_idx = np.clip(np.digitize(I_hl, i_edges) - 1, 0, density_bins - 1)
                q_idx = np.clip(np.digitize(Q_hl, q_edges) - 1, 0, density_bins - 1)
                # Look up density for each point (log scale for better visibility)
                density = np.log1p(hist[i_idx, q_idx])

                marker_config = dict(
                    size=marker_size * 1.5,
                    color=density,
                    colorscale=density_colorscale,
                    showscale=False,
                )
            else:
                marker_config = dict(
                    size=marker_size * 1.5,
                    color=highlight_color,
                )

            trace = go.Scatter3d(
                x=t_hl,
                y=I_hl,
                z=Q_hl,
                mode='markers',
                marker=marker_config,
                name='Highlighted' if ch == 0 else None,
                showlegend=(ch == 0 and not show_density),
            )
            if n_channels == 1:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=ch + 1, col=1)

        # Configure scene for this subplot
        scene_config = dict(
            xaxis_title='Time',
            yaxis_title='I',
            zaxis_title='Q',
            aspectmode='manual',
            aspectratio=dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2]),
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                up=dict(x=0, y=0, z=1),
            ),
            dragmode='orbit',  # Enable orbital rotation by default
        )
        fig.update_layout(**{scene_name: scene_config})

    # Configure overall layout
    top_margin = 40 if title else 10
    layout_config = dict(
        height=height * n_channels if n_channels > 1 else height,
        autosize=True,
        margin=dict(l=0, r=0, t=top_margin, b=0),
    )
    if title:
        layout_config['title'] = title
    if width is not None:
        layout_config['width'] = width
        layout_config['autosize'] = False
    fig.update_layout(**layout_config)

    return fig

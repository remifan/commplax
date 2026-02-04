# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "commplax[plot] @ git+https://github.com/remifan/commplax.git",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md("""
    # Basic MIMO Equalization

    This example demonstrates the core commplax workflow:
    1. Create an equalizer module
    2. Run it over a signal using `scan_with`
    3. Visualize the results

    Modules are [Equinox](https://github.com/patrick-kidger/equinox) PyTrees,
    compatible with all JAX transforms (jit, vmap, grad, etc.).
    """)
    return (mo,)


@app.cell
def _():
    import numpy as np
    from jax import numpy as jnp
    import matplotlib.pyplot as plt
    from scipy.signal import convolve
    from commplax import (
        module as mod,
        equalizer as eq,
        adaptive_filter as af,
        sym_map,
    )
    return af, convolve, eq, jnp, mod, np, plt, sym_map


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate Test Signal

    Create a dual-polarization 16-QAM signal and pass it through a simple channel model:
    - **SoP rotation**: polarization mixing via Jones matrix
    - **PMD**: 3-tap 2x2 MIMO filter emulating differential group delay
    """)
    return


@app.cell
def _(convolve, np, sym_map):
    def generate_test_signal(n_symbols=5000, snr_db=20, seed=42):
        """Generate 16-QAM signal through a 2x2 PMD channel."""
        np.random.seed(seed)
        const = sym_map.const('16QAM', norm=True)

        # Random symbols for 2 polarizations
        idx = np.random.randint(0, 16, size=(n_symbols, 2))
        tx = const[idx]

        # SoP rotation (Jones matrix)
        theta = 0.3
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        # PMD: 3-tap 2x2 MIMO channel (center-heavy)
        H = np.zeros((3, 2, 2), dtype=complex)
        H[0] = R @ np.array([[0.15, 0.05], [0.03, 0.12]])     # pre-cursor
        H[1] = R @ np.array([[0.85, 0.12], [0.08, 0.82]])     # main tap (center)
        H[2] = R @ np.array([[0.10, 0.03], [-0.02, 0.08]])    # post-cursor

        # Apply 2x2 MIMO convolution
        rx = np.zeros_like(tx)
        for i in range(2):
            for j in range(2):
                rx[:, i] += convolve(tx[:, j], H[:, i, j], mode='same')

        # Add noise
        noise_std = 10 ** (-snr_db / 20)
        rx += noise_std * (np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape))

        return rx

    rx_signal = generate_test_signal()
    print(f"RX signal shape: {rx_signal.shape}")
    return generate_test_signal, rx_signal


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Create MIMO Equalizer

    The `MIMOCell` is a 2x2 butterfly MIMO filter with adaptive taps.
    We use CMA (Constant Modulus Algorithm) for blind equalization.
    """)
    return


@app.cell
def _(af, eq):
    mimo = eq.MIMOCell(
        num_taps=5,
        af=af.rls_cma(),
        dims=2,
    )
    print(f"MIMO state shape: {mimo.state[0].shape}")
    return (mimo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run Equalization

    Use `scan_with` to run the equalizer over the signal.
    The module returns `(updated_module, output)`.
    """)
    return


@app.cell
def _(mimo, mod, rx_signal):
    mimo_out, eq_signal = mod.scan_with()(mimo, rx_signal)
    print(f"Output shape: {eq_signal.shape}")
    return eq_signal, mimo_out


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Visualize Results
    """)
    return


@app.cell
def _(eq_signal, plt, rx_signal):
    def plot_constellation(before, after, n_points=2000):
        """Plot before/after constellation comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, data, title in [
            (axes[0], before[-n_points:, 0], 'Before MIMO (X-pol)'),
            (axes[1], after[-n_points:, 0], 'After MIMO (X-pol)'),
        ]:
            ax.scatter(data.real, data.imag, s=1, alpha=0.5)
            ax.set(title=title, xlabel='I', ylabel='Q')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    _fig = plot_constellation(rx_signal, eq_signal)
    plt.gca()
    return (plot_constellation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Inspect Module State

    After running, the module contains updated filter taps.
    Modules are immutable - `scan_with` returns a new updated module.
    """)
    return


@app.cell
def _(mimo, mimo_out, mo):
    mo.ui.tabs({
        "Before": mo.md(mo.inspect(mimo, docs=False).text),
        "After": mo.md(mo.inspect(mimo_out, docs=False).text)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Filter Tap Visualization

    The 2x2 MIMO has 4 sub-filters: h_xx, h_xy, h_yx, h_yy.
    Compare tap magnitudes before (initialized) and after (converged).
    """)
    return


@app.cell
def _(mimo, mimo_out, np, plt):
    def plot_mimo_taps(taps_before, taps_after):
        """Plot 2x2 MIMO filter taps before/after comparison."""
        n_taps = taps_before.shape[-1]
        tap_idx = np.arange(n_taps)
        labels = [['h_xx (X→X)', 'h_xy (Y→X)'], ['h_yx (X→Y)', 'h_yy (Y→Y)']]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for row in range(2):
            for col in range(2):
                ax = axes[row, col]
                ax.stem(tap_idx - 0.1, np.abs(taps_before[row, col]),
                        linefmt='C0--', markerfmt='C0o', basefmt='C0-', label='Before')
                ax.stem(tap_idx + 0.1, np.abs(taps_after[row, col]),
                        linefmt='C1-', markerfmt='C1o', basefmt='C1-', label='After')
                ax.set(title=labels[row][col], xlabel='Tap index', ylabel='|h|')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    _fig = plot_mimo_taps(
        np.array(mimo.state[0]),
        np.array(mimo_out.state[0])
    )
    plt.gca()
    return (plot_mimo_taps,)


if __name__ == "__main__":
    app.run()

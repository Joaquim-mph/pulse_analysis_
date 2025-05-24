import scienceplots
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from typing import Optional
import numpy as np
from typing import Callable, Sequence, Optional, Tuple, Union, List, Dict
from eye_utils   import eye_diagram  
from styles import set_plot_style



    
def plot_pulse_markers(
    pulse_data: List[Tuple[str, ...]],
    prefix: str = "",
    show: bool = True,
    savefig: bool = True,
    which: str = "all",  # 'impulse', 'mag', 'db', or 'all'
    f_xlim: Tuple[float, float] = (-2, 2),
    f_mag_xlim: Optional[Tuple[float, float]] = None,
    f_db_xlim: Optional[Tuple[float, float]] = None,
    t_xlim: Tuple[float, float] = (-4, 4),
    figsize: Tuple[float, float] = (7, 4),
    markersize: int = 4,
    linewidth: float = 0.5,
    dpi: int = 300,
    legend_loc: str = 'best',
    T: float = 1.0,
    plot_negative: bool = True,
    markers: Optional[List[str]] = None,
    linestyles_list: Optional[List[str]] = None,
    db_ylim: Tuple[float, float] = (-160, 5),
    time_axis_label: str = r"$t/T$",
    freq_axis_label: str = r"$f/B$"
) -> Dict[str, plt.Figure]:
    """
    Plot comparison figures for pulse shapes in time and frequency domains.

    Parameters
    ----------
    pulse_data : list of tuples
        Each tuple: (label, t, h, f, mag, mag_db)
    prefix : str
        Prefix for saving figures (e.g., "figures/pulse").
    show : bool
        Whether to display the figures.
    savefig : bool
        Whether to save the figures to disk.
    which : str
        'impulse', 'mag', 'db', or 'all' — which plots to include.
    f_xlim : tuple
        Default x-axis limits for frequency plots.
    f_mag_xlim : tuple, optional
        x-axis limits for magnitude plot (overrides f_xlim).
    f_db_xlim : tuple, optional
        x-axis limits for dB plot (overrides f_xlim).
    t_xlim : tuple
        x-axis limits for time plots.
    figsize : tuple
        Size of each figure in inches.
    markersize : int
        Size of markers on plots.
    linewidth : float
        Width of plot lines.
    dpi : int
        Resolution of saved images.
    legend_loc : str
        Legend location (e.g., 'best', 'upper right').
    T : float
        Symbol period used for normalizing time axis.
    plot_negative : bool
        If True, include negative-time side of impulse.
    markers : list
        Custom list of markers to cycle through.
    linestyles_list : list
        Custom list of linestyles to cycle through.
    db_ylim : tuple
        y-axis limits for the dB plot.
    time_axis_label : str
        Label for time axis.
    freq_axis_label : str
        Label for frequency axis.

    Returns
    -------
    dict
        Dictionary with keys 'impulse', 'mag', and/or 'db' and their corresponding figures.
    """
    MARKERS = markers or ['D', 'o', '^', 's', '*', 'P', 'v', 'x']
    LINESTYLES = linestyles_list or ['-', '--', '-.', ':']
    figures = {}

    configs = {
        "impulse": {
            "title": "Impulse Response",
            "y_label": r"$h(t)$",
            "x_label": time_axis_label,
            "x_lim": t_xlim,
            "y_index": 2,
        },
        "mag": {
            "title": "Magnitude Spectrum",
            "y_label": r"$|H(f)|$",
            "x_label": freq_axis_label,
            "x_lim": f_mag_xlim if f_mag_xlim else f_xlim,
            "y_index": 4,
        },
        "db": {
            "title": "dB Spectrum",
            "y_label": "dB",
            "x_label": freq_axis_label,
            "x_lim": f_db_xlim if f_db_xlim else f_xlim,
            "y_index": 5,
        }
    }

    targets = ["impulse", "mag", "db"] if which == "all" else [which]

    for key in targets:
        cfg = configs[key]
        fig, ax = plt.subplots(figsize=figsize)

        for i, pulse in enumerate(pulse_data):
            label, t, h, f, mag, mag_db = pulse
            x = t / T if key == "impulse" else f
            y = pulse[cfg["y_index"]]
            pos = slice(None) if (key != "impulse" or plot_negative) else (x > 0)

            marker = MARKERS[i % len(MARKERS)]
            linestyle = LINESTYLES[i % len(LINESTYLES)]
            if key=="impulse":
                ax.plot(x[pos], y[pos], label=label, marker=marker,
                        linestyle=linestyle, markersize=markersize, linewidth=linewidth, markevery=2)
            else:
                ax.plot(x[pos], y[pos], label=label, marker=marker,
                        linestyle=linestyle, markersize=markersize, linewidth=linewidth, markevery=3)

        ax.set_title(cfg["title"])
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel(cfg["y_label"])
        ax.set_xlim(cfg["x_lim"])
        if key == "db" and db_ylim:
            ax.set_ylim(db_ylim)
        ax.legend(loc=legend_loc)
        ax.grid(True)
        fig.tight_layout()

        if savefig and prefix:
            os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
            fig.savefig(f"{prefix}_{key}.png", dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figures[key] = fig

    return figures




def save_figure(fig: plt.Figure, prefix: str, part: str, dpi: int):
    """
    Helper to save figure with proper directory creation.
    """
    path = f"{prefix}_eye_{part}.png"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)



def plot_eye_traces(
    eye_data: Optional[np.ndarray] = None,
    t_eye: Optional[np.ndarray] = None,
    pulse: Union[str, Callable] = None,
    *,
    eye_T: float = 2.0,
    parts: Sequence[str] = ("real",),
    fs: int = 10,
    prefix: str = "",
    show: bool = True,
    savefig: bool = True,
    figsize: Tuple[float, float] = (7, 7),
    linewidth: float = 0.2,
    color: str = "k",
    alpha: float = 0.3,
    dpi: int = 300,
    y_lim: Tuple[float, float] = (-2.5, 2.5),
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, plt.Figure, np.ndarray]:
    """
    Plot eye diagram using precomputed eye_data, or compute internally.

    Returns
    -------
    eye, t_eye, fig, axes
    """
    # Compute if necessary
    if eye_data is None or t_eye is None:
        eye_data, t_eye, _, _ = eye_diagram(pulse, fs=fs, eye_T=eye_T, **kwargs)

    n_parts = len(parts)
    fig, axes = plt.subplots(
        n_parts, 1,
        figsize=(figsize[0], figsize[1] * n_parts),
        sharex=True
    )
    axes = np.atleast_1d(axes)

    for ax, part in zip(axes, parts):
        if part not in {"real", "imag"}:
            raise ValueError("parts must be 'real' or 'imag'")
        data = (eye_data.real if part == "real" else eye_data.imag)
        ax.plot(t_eye, data.T, color=color, lw=linewidth, alpha=alpha)
        ax.set_title(f"Eye ({part}) — {pulse if isinstance(pulse, str) else pulse.__name__}")
        ax.set_xlabel("t / T")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-eye_T/2, eye_T/2)
        ax.set_ylim(y_lim)
        ax.grid(True)

    fig.tight_layout()
    if savefig and prefix:
        for part in parts:
            save_figure(fig, prefix, part, dpi)
    if show:
        plt.show()

    return eye_data, t_eye, fig, axes
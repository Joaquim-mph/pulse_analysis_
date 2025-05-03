import scienceplots
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from typing import Literal, Sequence, Optional, Tuple, Union, List, Dict
from eye_utils   import eye_diagram  

from pulse_toolbox import t_axis, compute_pulse, spectrum
from styles import set_plot_style


def init_pulse_plots(figsize=(25, 7), f_xlim=(-2, 2)):
    """
    Create 3-panel pulse plot (impulse, magnitude, dB magnitude).
    f_xlim: tuple, frequency axis limits for |H(f)| and dB
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].set(xlabel=r"$t/T$", ylabel=r"$h(t)$")
    axs[1].set(xlabel=r"$f$ (normalized)", ylabel=r"$|H(f)|$", xlim=f_xlim)
    axs[2].set(xlabel=r"$f$ (normalized)", ylabel="dB", xlim=f_xlim)

    for ax in axs:
        ax.grid(True)

    return fig, axs



def add_pulse_to_plot(t, h, f, mag, mag_db, axs, label: Optional[str] = None):
    """
    Plot a single pulse into existing figure/axes.
    """
    pos = t > 0
    axs[0].plot(t[pos], h[pos], label=label)
    axs[1].plot(f, mag, label=label)
    axs[2].plot(f, mag_db, label=label)


def finalize_pulse_plot(axs, show=True, save_path: Optional[str] = None):
    """
    Final layout tweaks and optional saving.
    """
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

    
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
            ax.plot(x[pos], y[pos], label=label, marker=marker,
                    linestyle=linestyle, markersize=markersize, linewidth=linewidth)

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


def compare_single_pulse_variants(
    pulse_name: str,
    param_list: List[Dict],
    span_T: float = 9,
    oversample: int = 200,
    T: float = 1.0,
    normalize: str = "amplitude",
    norm_label: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (13, 5),
    title: Optional[str] = None,
    spectrum_oversize: int = 2
) -> None:

    os.makedirs(os.path.dirname(save_path or "figures"), exist_ok=True)
    t, dt = t_axis(span_T=span_T, oversample=oversample, T=T)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].set_title("Time-Domain Pulse")
    axs[0].set_xlabel("$t/T$")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    axs[0].set_xlim(-2, 2)

    axs[1].set_title("Spectrum $|H(f)|$")
    axs[1].set_xlabel("$f/B$")
    axs[1].set_ylabel("Magnitude")
    axs[1].grid(True)
    axs[1].set_xlim(-2, 2)
    
    latex_symbols = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "epsilon": r"\epsilon",
        "mu": r"\mu"
    }

    for params in param_list:
        label = ", ".join([
            rf"${latex_symbols.get(key, key)}$={value}" for key, value in params.items()
            if isinstance(value, (int, float))
        ])

        alpha = params.get("alpha", 0.35)
        pulse_args = {k: v for k, v in params.items() if k != "alpha"}

        h = compute_pulse(t, pulse_name, alpha=alpha,
                          T=T, dt=dt, normalize=normalize, **pulse_args)
        axs[0].plot(t, h, label=label)

        spec = spectrum(h, fs=1/dt, nfft=oversample * spectrum_oversize * span_T * 2)
        axs[1].plot(spec["fB"], spec["mag"], label=label)

    axs[0].legend(fontsize=9)
    axs[1].legend(fontsize=9)
    title_str = title or f"{pulse_name} Variants Comparison"
    if norm_label:
        title_str += f" ({norm_label})"
    fig.suptitle(title_str, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def compare_multiple_pulses(
    pulse_variants: List[Tuple[str, List[Dict]]],
    normalize: str = "amplitude",
    norm_label: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = "Pulse Family Comparison",
    span_T: float = 9,
    oversample: int = 200,
    T: float = 1.0,
    figsize: Tuple[int, int] = (14, 6),
    spectrum_oversize: int = 2
) -> None:

    os.makedirs(os.path.dirname(save_path or "figures"), exist_ok=True)
    t, dt = t_axis(span_T=span_T, oversample=oversample, T=T)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].set_title("Time-Domain Pulse")
    axs[0].set_xlabel("$t/T$")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    axs[0].set_xlim(-2, 2)

    axs[1].set_title("Spectrum $|H(f)|$")
    axs[1].set_xlabel("$f/B$")
    axs[1].set_ylabel("Magnitude")
    axs[1].grid(True)
    axs[1].set_xlim(-2, 2)
    
    
    latex_symbols = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "epsilon": r"\epsilon",
        "mu": r"\mu"
    }

    for pulse_name, param_list in pulse_variants:
        for params in param_list:
            label = f"{pulse_name}: " + ", ".join([
                rf"${latex_symbols.get(key, key)}$={value}" for key, value in params.items()
                if isinstance(value, (int, float))
            ])

            alpha = params.get("alpha", 0.35)
            pulse_args = {k: v for k, v in params.items() if k != "alpha"}

            h = compute_pulse(t, pulse_name, alpha=alpha, T=T, dt=dt, normalize=normalize, **pulse_args)
            axs[0].plot(t, h, label=label)

            spec = spectrum(h, fs=1/dt, nfft=oversample * spectrum_oversize * span_T * 2)
            axs[1].plot(spec["fB"], spec["mag"], label=label)

    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    if norm_label:
        title += f" ({norm_label})"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()



def plot_eye_traces(
    pulse: Union[str, callable],
    *,
    alpha: float = 0.35,
    fs: int = 10,
    span_T: float = 6,
    n_symbols: int = 100_000,
    eye_T: float = 2.0,
    max_traces: int = 500,
    pulse_kwargs: Optional[dict] = None,
    parts: Sequence[str] = ("real",),
    style: str = "default",
    prefix: str = "",
    show: bool = True,
    savefig: bool = True,
    figsize: Tuple[float, float] = (7, 7),
    linewidth: float = 0.1,
    color: str = "k",
    dpi: int = 300,
    y_lim: Tuple[float, float] = (-2.5, 2.5),
    normalize: Literal["amplitude", "continuous", "discrete"] = "continuous"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and plot a highly‑customisable eye diagram.

    Parameters
    ----------
    pulse, alpha, fs, span_T, n_symbols, eye_T, max_traces, pulse_kwargs
        Same semantics as in `eye_diagram()` (see its docstring).
    parts : ('real' | 'imag')[...]
        Components to plot.  Use both to stack two subplots.
    style : str
        Key in your STYLE_CONFIGS dict (default 'default').
    prefix : str
        If non‑empty, figures saved as  f"{prefix}_eye_[part].png" .
    show : bool
        Call `plt.show()`.  If False figure is returned but not shown.
    savefig : bool
        Save to disk.
    figsize : (float, float)
        Single‑subplot figure size; if two parts are requested the height
        is doubled automatically.
    linewidth, color
        Trace style.
    dpi : int
        Save resolution.
    y_lim : (ymin, ymax)
        y‑axis limits.
    normalize : str
        Pulse normalization method: 'amplitude', 'continuous', or 'discrete'.

    Returns
    -------
    eye : np.ndarray
        Eye matrix (n_traces × eye_T*fs).
    t_eye : np.ndarray
        x‑axis for one eye trace.
    """
    # ------------------------------------------------------------------ data
    eye, t_eye = eye_diagram(
        pulse, alpha=alpha, fs=fs, span_T=span_T,
        n_symbols=n_symbols, eye_T=eye_T, max_traces=max_traces,
        pulse_kwargs=pulse_kwargs, parts=parts, show=False,
        normalize=normalize
    )

    # ------------------------------------------------------------------ plot
    set_plot_style(style)
    n_parts = len(parts)
    fig, axes = plt.subplots(
        n_parts, 1, figsize=(figsize[0], figsize[1] * n_parts),
        sharex=True
    )
    axes = np.atleast_1d(axes)

    for ax, part in zip(axes, parts):
        if part not in {"real", "imag"}:
            raise ValueError("parts must be 'real' or 'imag'")
        data = eye.real if part == "real" else eye.imag
        ax.plot(t_eye, data.T, color=color, lw=linewidth)
        ax.set(
            title=f"Eye ({part}) — {pulse if isinstance(pulse,str) else pulse.__name__}",
            xlabel="t / T",
            ylabel="Amplitude",
            xlim=(-eye_T/2, eye_T/2),
            ylim=y_lim
        )
        ax.grid(True)

    fig.tight_layout()
    if savefig and prefix:
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        for part, ax in zip(parts, axes):
            fig_path = f"{prefix}_eye_{part}.png"
            fig.savefig(fig_path, dpi=dpi)
    if show:
        plt.show()

    return eye, t_eye
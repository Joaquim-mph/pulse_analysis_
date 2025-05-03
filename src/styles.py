import matplotlib.pyplot as plt 
# Common settings to apply to all styles
common_rc = {
    "text.usetex": False,
    "lines.markersize": 6,
    "legend.fontsize": 12,
    "axes.grid": True,
}

STYLE_CONFIGS = {
    "default": {
        "base": ["science", "nature"],
        "rc": {
            **common_rc,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "legend.frameon": True,
            "axes.prop_cycle": plt.cycler(color=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
                '#d62728', '#8c564b', '#e377c2', '#7f7f7f',
                '#17becf', '#bcbd22'
            ]),
        }
    },

    "solar_flare": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.edgecolor": "#880000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "grid.color": "#ddcccc",
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "legend.frameon": False,
            "figure.figsize": (8.5, 4.8),
            "axes.prop_cycle": plt.cycler(color=[
                "#e63946", "#ff7f50", "#ffa600", "#bc5090",
                "#ff6361", "#d45087", "#f95d6a", "#b81d24"
            ]),
        }
    },

    "solar_flare_dark": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.edgecolor": "#ff6666",
            "axes.labelcolor": "#ffffff",
            "xtick.color": "#dddddd",
            "ytick.color": "#dddddd",
            "grid.color": "#772222",
            "grid.linestyle": "--",
            "grid.linewidth": 0.4,
            "lines.linewidth": 2.0,
            "legend.labelcolor": "#ffffff",
            "legend.frameon": False,
            "figure.figsize": (8.5, 4.8),
            "axes.prop_cycle": plt.cycler(color=[
                "#ff6b6b", "#ff9966", "#ffcc99", "#ff4d6d",
                "#ffb347", "#ee6055", "#ffd6a5", "#ff5c57"
            ]),
        }
    },

    "nova": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "font.family": "serif",
            "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.5,
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1,
            "legend.frameon": False,
            "figure.figsize": (8, 4.5),
            "axes.prop_cycle": plt.cycler(color=[
                "#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#F0E442", "#56B4E9", "#E69F00", "#999999"
            ]),
        }
    },

    "dark_nova": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#dddddd",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "text.color": "#ffffff",
            "grid.color": "#555555",
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.labelcolor": "#dddddd",
            "figure.figsize": (8, 4.5),
            "axes.prop_cycle": plt.cycler(color=[
                "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                "#0072B2", "#D55E00", "#CC79A7", "#ffffff"
            ]),
        }
    },

    "super_nova": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1,
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "legend.loc": "best",
            "legend.labelspacing": 0.3,
            "figure.figsize": (8.5, 4.8),
            "axes.prop_cycle": plt.cycler(color=[
                "#ff6f61", "#6baed6", "#50c878", "#bc80bd",
                "#fdb462", "#8dd3c7", "#c0c0c0", "#d95f02"
            ]),
        }
    },

    "super_nova_dark": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "figure.facecolor": "#0e0e0e",
            "axes.facecolor": "#0e0e0e",
            "savefig.facecolor": "#0e0e0e",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.edgecolor": "#dddddd",
            "axes.labelcolor": "#ffffff",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "grid.color": "#444444",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "legend.loc": "best",
            "legend.labelcolor": "#ffffff",
            "figure.figsize": (8.5, 4.8),
            "axes.prop_cycle": plt.cycler(color=[
                "#ff6f61", "#6baed6", "#50c878", "#bc80bd",
                "#fdb462", "#8dd3c7", "#e5e5e5", "#d95f02"
            ]),
        }
    },

    "matlab": {
        "base": ["science"],
        "rc": {
            **common_rc,
            "font.family": "serif",
            "font.size": 11,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "axes.prop_cycle": plt.cycler(color=["black"]),
        }
    }
}




def set_plot_style(style: str = "default"):
    from matplotlib import rcParams
    from matplotlib import pyplot as plt

    if style not in STYLE_CONFIGS:
        raise ValueError(f"Unknown style '{style}'.")

    config = STYLE_CONFIGS[style]

    # 1. Apply base style(s) like 'science'
    for base in config.get("base", []):
        plt.style.use(base)

    # 2. Then override rcParams
    for key, value in config.get("rc", {}).items():
        rcParams[key] = value

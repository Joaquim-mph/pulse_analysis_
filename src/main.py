from pulse_toolbox import get_pulse_info
from plot_utils import (
    plot_pulse_markers,
    plot_eye_traces
)
from styles import set_plot_style
from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)


# ──────────────────────────────────────────────────────────────
# 1. Style setup
set_plot_style("prism_rain")

# ──────────────────────────────────────────────────────────────
# 2. Global parameters
span_T   = 10
ovs      = 20
nfft     = 2048
T        = 1.0
alpha    = 0.35
normalize = "amplitude"
freq_axis = "fB"  # Can be "fT" or "fB"


# ──────────────────────────────────────────────────────────────
# 3. Generate pulses (amplitude‑normalized)

pulse_specs = [
    ("Raised Cosine", "raised_cosine", dict()),
    ("BTRC", "btrc", dict()),
    ("ELP   β=0.1", "elp", dict(beta=0.1)),
    ("IPLCP μ=1.6  ε=0.1", "iplcp", dict(mu=1.6, gamma=1, epsilon=0.1)),
]

pulse_data_022 = []
for label, key, extra in pulse_specs:
    info = get_pulse_info(
        key, 0.22, span_T,
        T=T, oversample=ovs,
        nfft=nfft,
        normalize=normalize,
        freq_axis=freq_axis,
        **extra
    )
    pulse_data_022.append((
        label,
        info["t"],       # Normalize time axis
        info["h"],
        info["f"],           # Frequency axis: either f/T or f/B
        info["mag"],
        info["mag_db"]
    ))



plot_pulse_markers(
    pulse_data_022,
    t_xlim=(0, 4),
    f_xlim=(0,5),
    f_mag_xlim=(0,3),
    prefix="figures/pulse_compare_022",
    show=False,
    figsize=(7, 7),
    markersize=3,
    linewidth=0.7,
    db_ylim=(-200,5),
    freq_axis_label="f/B",
    f_db_xlim=(-10,10)
)


pulse_data_05 = []
for label, key, extra in pulse_specs:
    info = get_pulse_info(
        key, 0.5, span_T,
        T=T, oversample=ovs,
        nfft=nfft,
        normalize=normalize,
        freq_axis=freq_axis,
        **extra
    )
    pulse_data_05.append((
        label,
        info["t"],       # Normalize time axis
        info["h"],
        info["f"],           # Frequency axis: either f/T or f/B
        info["mag"],
        info["mag_db"]
    ))



plot_pulse_markers(
    pulse_data_05,
    t_xlim=(0, 4),
    f_xlim=(0,5),
    f_mag_xlim=(0,3),
    prefix="figures/pulse_compare_05",
    show=False,
    figsize=(7, 7),
    markersize=3,
    linewidth=0.7,
    db_ylim=(-200,5),
    freq_axis_label="f/B",
    f_db_xlim=(-10,10)
)


# ──────────────────────────────────────────────────────────────
# 4. Closed‑form BER for Raised‑Cosine
snr_db = 10.0
offsets = (0.05, 0.10, 0.20, 0.25)
ber_cf = ber_isi_closed_form(
    pulse="raised_cosine",
    alpha=alpha,
    snr_db=snr_db,
    offsets=offsets,
)
print("Craig BER (Raised Cosine):", ber_cf)



    
# ──────────────────────────────────────────────────────────────
# 6. Eye Diagrams

set_plot_style("prism_rain")


_, _, rc_max = plot_eye_traces(
    "raised_cosine",
    alpha=0.22,
    normalize="continuous",
    parts=("real",),
    prefix="figures/rc_energy",
    show=False
)

_, _, btrc_max = plot_eye_traces(
    "btrc",
    alpha=0.22,
    normalize="continuous",
    parts=("real",),
    prefix="figures/btrc_energy",
    show=False
)

_, _, elp_max = plot_eye_traces(
    "elp",
    alpha=0.22,
    pulse_kwargs=dict(beta=0.1),
    normalize="continuous",
    parts=("real",),
    prefix="figures/elp_energy",
    show=False
)


_, _, iplcp_max = plot_eye_traces(
    "iplcp",
    alpha=0.22,
    pulse_kwargs=dict(mu=1.6, gamma=1, epsilon=0.1),
    normalize="continuous",
    parts=("real",),
    prefix="figures/iplcp_energy",
    show=False
)

print("\n Maximum Amplitude Values from Eye Diagrams (α = 0.22):\n")
print(f"{'Pulse':<10} | {'Max Amplitude':>15}")
print("-" * 30)
print(f"{'RC':<10} | {rc_max:>15.4f}")
print(f"{'BTRC':<10} | {btrc_max:>15.4f}")
print(f"{'ELP':<10} | {elp_max:>15.4f}")
print(f"{'IPLCP':<10} | {iplcp_max:>15.4f}")

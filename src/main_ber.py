from itertools import product
from functools import partial
from pathlib import Path

from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)
from pulse_table_utils import (
    truncate_pulse,
    results_to_df,
    latex_table,
    save_df_to_csv
)

# ----------------------------------------
# 1) Global parameters
# ----------------------------------------
snr_values   = [10, 15]
sir_values   = [10, 20]
alpha_values = [0.22, 0.35, 0.50]
L_values     = [2, 6]
offsets      = (0.05, 0.10, 0.20, 0.25)
pulses       = ["raised_cosine", "btrc", "elp", "iplcp"]
nbits, M, omega = 2**10, 100, 0.10
trunc_limits = [5.0, 10.0]

pulse_kwargs = {
    "raised_cosine": {},
    "btrc":         {},
    "elp":          {"beta": 0.1},
    "iplcp":        {"mu": 1.6, "gamma": 1, "epsilon": 0.1},
}

# ----------------------------------------
# 2) Resolve pulses once
# ----------------------------------------
resolved = {
    name: partial(_resolve_pulse(name), **pulse_kwargs.get(name, {}))
    for name in pulses
}

# ----------------------------------------
# 3) Generic BER runner
# ----------------------------------------
def run_ber(func, config_list, **common_kwargs):
    return {key: func(**params, **common_kwargs) for key, params in config_list}

# ----------------------------------------
# 4) Build configurations
# ----------------------------------------
# ISI by SNR
isi_configs = {
    snr: [
        (
            f"{pulse}_SNR{snr}_alpha{alpha}",
            {"pulse": resolved[pulse], "snr_db": snr, "alpha": alpha, "nbits": nbits}
        )
        for pulse, alpha in product(pulses, alpha_values)
    ]
    for snr in snr_values
}

# CCI by SIR and L
cci_configs = {
    (sir, L): [
        (
            f"{pulse}_SNR15_SIR{sir}_alpha{alpha}_L{L}",
            {"pulse": resolved[pulse], "snr_db": 15, "sir_db": sir, "alpha": alpha, "L": L}
        )
        for pulse, alpha in product(pulses, alpha_values)
    ]
    for sir, L in product(sir_values, L_values)
}

# Joint ISI+CCI
isi_cci_config = [
    (
        f"{pulse}_SNR15_SIR15_alpha{alpha}_L6_joint",
        {"pulse": resolved[pulse], "snr_db": 15, "sir_db": 15, "alpha": alpha, "L": 6, "nbits": nbits}
    )
    for pulse, alpha in product(pulses, alpha_values)
]

# ISI truncated by T
isi_trunc_configs = {
    int(T): [
        (
            f"{pulse}_SNR10_alpha{alpha}_trunc{int(T)}",
            {"pulse": truncate_pulse(resolved[pulse], T), "snr_db": 10, "alpha": alpha, "nbits": nbits}
        )
        for pulse, alpha in product(pulses, alpha_values)
    ]
    for T in trunc_limits
}

# CCI truncated by T (fixed SNR=15, SIR=10, L=2)
cci_trunc_configs = {
    int(T): [
        (
            f"{pulse}_SNR15_SIR10_alpha{alpha}_L2_trunc{int(T)}",
            {"pulse": truncate_pulse(resolved[pulse], T), "snr_db": 15, "sir_db": 10, "alpha": alpha, "L": 2}
        )
        for pulse, alpha in product(pulses, alpha_values)
    ]
    for T in trunc_limits
}

# Joint ISI+CCI truncated for alpha=0.22
isi_cci_trunc_config = [
    (
        f"{pulse}_SNR15_SIR15_alpha0.22_L6_joint_trunc{int(T)}",
        {"pulse": truncate_pulse(resolved[pulse], T), "snr_db": 15, "sir_db": 15, "alpha": 0.22, "L": 6, "nbits": nbits}
    )
    for T in trunc_limits
    for pulse in pulses
]

# ----------------------------------------
# 5) Run all experiments
# ----------------------------------------
common_kwargs = {"offsets": offsets, "M": M, "omega": omega}

isi_results      = {snr: run_ber(ber_isi_closed_form, cfgs, **common_kwargs)
                    for snr, cfgs in isi_configs.items()}
cci_results      = {(sir, L): run_ber(ber_cci_closed_form, cfgs, **common_kwargs)
                    for (sir, L), cfgs in cci_configs.items()}
isi_cci_results  = run_ber(ber_cci_isi_closed_form, isi_cci_config, **common_kwargs)
isi_trunc_results= {T: run_ber(ber_isi_closed_form, cfgs, **common_kwargs)
                    for T, cfgs in isi_trunc_configs.items()}
cci_trunc_results= {T: run_ber(ber_cci_closed_form, cfgs, **common_kwargs)
                    for T, cfgs in cci_trunc_configs.items()}
isi_cci_trunc    = run_ber(ber_cci_isi_closed_form, isi_cci_trunc_config, **common_kwargs)

# ----------------------------------------
# 6) Export: LaTeX tables and CSVs
# ----------------------------------------
# ISI tables
for snr, results in isi_results.items():
    df = results_to_df(results)
    print(f"\n% ——— BER ISI @ SNR={snr} dB ———")
    latex_table(df,
                caption=f"BER ISI Results (SNR = {snr} dB)",
                label=f"tab:ber_isi_snr{snr}")

# CCI tables
for (sir, L), results in cci_results.items():
    df = results_to_df(results)
    print(f"\n% ——— BER CCI @ SIR={sir} dB, L={L} ———")
    latex_table(df,
                caption=f"BER CCI Results (SIR = {sir} dB, L = {L})",
                label=f"tab:ber_cci_sir{sir}_L{L}")

# ISI+CCI
print()
latex_table(results_to_df(isi_cci_results),
            caption="BER ISI+CCI Results",
            label="tab:ber_isi_cci")

# Truncated ISI
for T, results in isi_trunc_results.items():
    df = results_to_df(results)
    print(f"\n% ——— BER ISI Truncated @ T={T} ———")
    latex_table(df,
                caption=f"BER ISI Results Truncated at T = {T}",
                label=f"tab:ber_isi_trunc{T}")

# Truncated CCI
for T, results in cci_trunc_results.items():
    df = results_to_df(results)
    print(f"\n% ——— BER CCI Truncated @ T={T} ———")
    latex_table(df,
                caption=f"BER CCI Results Truncated at T = {T}",
                label=f"tab:ber_cci_trunc{T}")

# Truncated ISI+CCI
print()
latex_table(results_to_df(isi_cci_trunc),
            caption="BER ISI+CCI Truncated Results",
            label="tab:ber_isi_cci_truncated")


# ----------------------------------------
# 7) Save CSVs for further analysis
# ----------------------------------------
# This folder will be created by save_df_to_csv if it doesn't exist
results_folder = "results"

# ISI per SNR or snr, results in isi_results.items():
df = results_to_df(results)
save_df_to_csv(df, f"isi_snr{snr}.csv", folder=results_folder)

# CCI per (SIR, L)
for (sir, L), results in cci_results.items():
    df = results_to_df(results)
    save_df_to_csv(df, f"cci_sir{sir}_L{L}.csv", folder=results_folder)

# ISI+CCI

df_isi_cci = results_to_df(isi_cci_results)
save_df_to_csv(df_isi_cci, "isi_cci.csv", folder=results_folder)

# Truncated ISI per T
for T, results in isi_trunc_results.items():
    df = results_to_df(results)
    save_df_to_csv(df, f"isi_trunc{T}.csv", folder=results_folder)

# Truncated CCI per T
for T, results in cci_trunc_results.items():
    df = results_to_df(results)
    save_df_to_csv(df, f"cci_trunc{T}.csv", folder=results_folder)

# Truncated ISI+CCI

df_isi_cci_trunc = results_to_df(isi_cci_trunc)
save_df_to_csv(df_isi_cci_trunc, "isi_cci_trunc.csv", folder=results_folder)

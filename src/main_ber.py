from itertools import product
from functools import partial
from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)
from pulse_table_utils import truncate_pulse, results_to_df, latex_table, save_df_to_csv 

# 1) global parameters
snr_values     = [10.0, 15.0]
sir_values     = [10.0, 20.0]
alpha_values   = [0.22, 0.35, 0.50]
L_values       = [2, 6]
offsets        = (0.05, 0.10, 0.20, 0.25)
pulse_list     = ["raised_cosine", "btrc", "elp", "iplcp"]
nbits, M, omega = 2**10, 100, 0.10
trunc_limits   = [5.0, 10.0]

pulse_kwargs = {
    "raised_cosine": {},
    "btrc":         {},
    "elp":          {"beta": 0.1},
    "iplcp":        {"mu": 1.6, "gamma": 1, "epsilon": 0.1},
}

# 2) pre-resolve each pulse once
resolved_pulses = {
    name: (lambda p, kw: partial(p, **kw))(
        _resolve_pulse(name),
        pulse_kwargs.get(name, {})
    )
    for name in pulse_list
}

# 3) generic runner
def run_ber(func, key_params, **shared_kwargs):
    return {
        key: func(**params, **shared_kwargs)
        for key, params in key_params
    }

# 4) build all the (key, params) lists
isi_keys = [
    (
      f"{pulse}_SNR{snr}_alpha{alpha}",
      dict(pulse=resolved_pulses[pulse], snr_db=snr, alpha=alpha, nbits=nbits)
    )
    for pulse, snr, alpha in product(pulse_list, snr_values, alpha_values)
]

cci_keys = [
    (
      f"{pulse}_SNR{15}_SIR{sir}_alpha{alpha}_L{L}",
      dict(pulse=resolved_pulses[pulse], snr_db=15.0, sir_db=sir, alpha=alpha, L=L)
    )
    for pulse, sir, alpha, L in product(pulse_list, sir_values, alpha_values, L_values)
]

isi_cci_keys = [
    (
      f"{pulse}_SNR{15}_SIR{15}_alpha{alpha}_L{6}_joint",
      dict(pulse=resolved_pulses[pulse], snr_db=15.0, sir_db=15.0,
           alpha=alpha, L=6, nbits=nbits)
    )
    for pulse, alpha in product(pulse_list, alpha_values)
]

isi_trunc_keys = [
    (
      f"{pulse}_SNR{10}_alpha{alpha}_trunc{int(T)}",
      dict(pulse=truncate_pulse(resolved_pulses[pulse], T),
           snr_db=10.0, alpha=alpha, nbits=nbits)
    )
    for T in trunc_limits
    for pulse, alpha in product(pulse_list, alpha_values)
]

cci_trunc_keys = [
    (
      f"{pulse}_SNR{15}_SIR{10}_alpha{alpha}_L{2}_trunc{int(T)}",
      dict(pulse=truncate_pulse(resolved_pulses[pulse], T),
           snr_db=15.0, sir_db=10.0, alpha=alpha, L=2)
    )
    for T in trunc_limits
    for pulse, alpha in product(pulse_list, alpha_values)
]

isi_cci_trunc_keys = [
    (
      f"{pulse}_SNR{15}_SIR{15}_alpha{alpha}_L{6}_joint_trunc{int(T)}",
      dict(pulse=truncate_pulse(resolved_pulses[pulse], T),
           snr_db=15.0, sir_db=15.0, alpha=alpha, L=6, nbits=nbits)
    )
    for T in trunc_limits
    for pulse, alpha in product(pulse_list, [0.22])   # only alpha=0.22
]

# 5) actually run them
common = dict(offsets=offsets, M=M, omega=omega)
isi_results           = run_ber(ber_isi_closed_form,      isi_keys,         **common)
cci_results           = run_ber(ber_cci_closed_form,      cci_keys,         **common)
isi_cci_results       = run_ber(ber_cci_isi_closed_form, isi_cci_keys,     **common)
isi_trunc_results     = run_ber(ber_isi_closed_form,      isi_trunc_keys,   **common)
cci_trunc_results     = run_ber(ber_cci_closed_form,      cci_trunc_keys,   **common)
isi_cci_trunc_results = run_ber(ber_cci_isi_closed_form, isi_cci_trunc_keys, **common)

# 6) export as before
df_isi               = results_to_df(isi_results)
df_cci               = results_to_df(cci_results)
df_isi_cci           = results_to_df(isi_cci_results)
df_isi_truncated     = results_to_df(isi_trunc_results)
df_cci_truncated     = results_to_df(cci_trunc_results)
df_isi_cci_truncated = results_to_df(isi_cci_trunc_results)

print()
latex_table(
    df_isi,
    caption="BER ISI Results",
    label="tab:ber_isi"
)
print()

latex_table(
    df_cci,
    caption="BER CCI Results",
    label="tab:ber_cci"
)
print()

latex_table(
    df_isi_cci,
    caption="BER ISI+CCI Results",
    label="tab:ber_isi_cci"
)
print()

latex_table(
    df_isi_truncated,
    caption="BER ISI Truncated Results",
    label="tab:ber_isi_truncated"
)
print()

latex_table(
    df_cci_truncated,
    caption="BER CCI Truncated Results",
    label="tab:ber_cci_truncated"
)
print()

latex_table(
    df_isi_cci_truncated,
    caption="BER ISI+CCI Truncated Results",
    label="tab:ber_isi_cci_truncated"
)




save_df_to_csv(df_isi, "isi.csv")
save_df_to_csv(df_cci, "cci.csv")
save_df_to_csv(df_isi, "isi_cci.csv")

save_df_to_csv(df_isi_truncated, "isi_truncated,.csv")
save_df_to_csv(df_cci_truncated, "cci_truncated,.csv")
save_df_to_csv(df_isi_cci_truncated, "isi_cci_truncated,.csv")










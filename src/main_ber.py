from itertools import product
from functools import partial
import numpy as np
from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)
from latex_utils import (
    export_flat_latex_table,
    export_cci_latex_table,
    export_joint_latex_table,
    truncate_pulse,
    export_cci_latex_table_truncated,
    export_flat_latex_table_truncated,
    export_joint_latex_table_truncated
)
from pulse_table_utils import results_to_df

# ─────────────────────────────────────────────────────────────
# 2. Parámetros globales y funciones registradas
# ─────────────────────────────────────────────────────────────

snr_values = [10.0, 15.0]
alpha_values = [0.22, 0.35, 0.50]
offsets = (0.05, 0.10, 0.20, 0.25)
pulse_list = ["raised_cosine", "btrc", "elp", "iplcp"]

nbits = 2**10  # 1024 símbolos
M = 100
omega = 0.10

pulse_kwargs_dict = {
    "raised_cosine": {},
    "btrc": {},
    "elp": {"beta": 0.1},
    "iplcp": {"mu": 1.6, "gamma": 1, "epsilon": 0.1},
}

# ─────────────────────────────────────────────────────────────
# 3. Cálculo de BER: ISI
# ─────────────────────────────────────────────────────────────

isi_results = {}

for pulse, snr, alpha in product(pulse_list, snr_values, alpha_values):
    key = f"{pulse}_SNR{snr}_alpha{alpha}"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    isi_results[key] = ber_isi_closed_form(
        pulse=resolved_pulse,
        alpha=alpha,
        snr_db=snr,
        offsets=offsets,
        nbits=nbits,
        M=M,
        omega=omega
    )

# ─────────────────────────────────────────────────────────────
# 4. Cálculo de BER: CCI (SNR = 15 dB)
# ─────────────────────────────────────────────────────────────

sir_values = [10.0, 20.0]
L_values = [2, 6]
snr = 15.0

cci_results = {}

for pulse, sir, alpha, L in product(pulse_list, sir_values, alpha_values, L_values):
    key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    cci_results[key] = ber_cci_closed_form(
        pulse=resolved_pulse,
        alpha=alpha,
        snr_db=snr,
        sir_db=sir,
        L=L,
        offsets=offsets,
        M=M,
        omega=omega
    )

# ─────────────────────────────────────────────────────────────
# 5. Cálculo de BER: ISI + CCI (L = 6, SNR = SIR = 15 dB)
# ─────────────────────────────────────────────────────────────

sir = 15.0
L = 6
isi_cci_results = {}

for pulse, alpha in product(pulse_list, alpha_values):
    key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}_joint"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    isi_cci_results[key] = ber_cci_isi_closed_form(
        pulse=resolved_pulse,
        alpha=alpha,
        snr_db=snr,
        sir_db=sir,
        L=L,
        nbits=nbits,
        offsets=offsets,
        M=M,
        omega=omega
    )




isi_trunc_results = {}
snr = 10.0
truncation_limits = [5.0, 10.0]

for t_max in truncation_limits:
    for pulse, alpha in product(pulse_list, alpha_values):
        key = f"{pulse}_SNR{snr}_alpha{alpha}_trunc{int(t_max)}"
        base_pulse = _resolve_pulse(pulse)
        kwargs = pulse_kwargs_dict.get(pulse, {})
        resolved_base = partial(base_pulse, **kwargs) if kwargs else base_pulse
        resolved_pulse = truncate_pulse(resolved_base, t_max)

        isi_trunc_results[key] = ber_isi_closed_form(
            pulse=resolved_pulse,
            alpha=alpha,
            snr_db=snr,
            offsets=offsets,
            nbits=nbits,
            M=M,
            omega=omega
        )


# Solo para SNR=15 dB, SIR=10 dB y L=2
cci_trunc_results = {}
snr = 15.0
sir = 10.0
L = 2

for t_max in truncation_limits:
    for pulse, alpha in product(pulse_list, alpha_values):
        key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}_trunc{int(t_max)}"
        base_pulse = _resolve_pulse(pulse)
        kwargs     = pulse_kwargs_dict.get(pulse, {})
        resolved_base = partial(base_pulse, **kwargs) if kwargs else base_pulse
        resolved_pulse = truncate_pulse(resolved_base, t_max)

        cci_trunc_results[key] = ber_cci_closed_form(
            pulse    = resolved_pulse,
            alpha    = alpha,
            snr_db   = snr,
            sir_db   = sir,
            L        = L,
            offsets  = offsets,
            M        = M,
            omega    = omega
        )


isi_cci_trunc_results = {}
snr = 15.0
sir = 15.0
L = 6
# Only for alpha=0.22 
alpha_subset = [0.22]

for t_max in truncation_limits:
    for pulse, alpha in product(pulse_list, alpha_subset):
        key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}_joint_trunc{int(t_max)}"
        base_pulse = _resolve_pulse(pulse)
        kwargs = pulse_kwargs_dict.get(pulse, {})
        resolved_base = partial(base_pulse, **kwargs) if kwargs else base_pulse
        resolved_pulse = truncate_pulse(resolved_base, t_max)

        isi_cci_trunc_results[key] = ber_cci_isi_closed_form(
            pulse=resolved_pulse,
            alpha=alpha,
            snr_db=snr,
            sir_db=sir,
            L=L,
            nbits=nbits,
            offsets=offsets,
            M=M,
            omega=omega
        )


# ─────────────────────────────────────────────────────────────
# 6. Exportación de resultados a LaTeX
# ─────────────────────────────────────────────────────────────

df = results_to_df(isi_results)


print(df)


# export_flat_latex_table(isi_results)
# export_cci_latex_table(cci_results)
# export_joint_latex_table(isi_cci_results)

# export_flat_latex_table_truncated(isi_trunc_results)
# export_cci_latex_table_truncated(cci_trunc_results)
# export_joint_latex_table_truncated(isi_cci_trunc_results)
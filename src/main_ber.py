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

results = {}

for pulse, snr, alpha in product(pulse_list, snr_values, alpha_values):
    key = f"{pulse}_SNR{snr}_alpha{alpha}"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    results[key] = ber_isi_closed_form(
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

results_cci = {}

for pulse, sir, alpha, L in product(pulse_list, sir_values, alpha_values, L_values):
    key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    results_cci[key] = ber_cci_closed_form(
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
results_joint = {}

for pulse, alpha in product(pulse_list, alpha_values):
    key = f"{pulse}_SNR{snr}_SIR{sir}_alpha{alpha}_L{L}_joint"
    base_pulse = _resolve_pulse(pulse)
    kwargs = pulse_kwargs_dict.get(pulse, {})
    resolved_pulse = partial(base_pulse, **kwargs) if kwargs else base_pulse

    results_joint[key] = ber_cci_isi_closed_form(
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

print("\n\n% === ISI ===\n")
export_flat_latex_table(results)

print("\n\n% === CCI ===\n")
export_cci_latex_table(results_cci)

print("\n\n% === ISI + CCI ===\n")
export_joint_latex_table(results_joint)






results = {}
snr = 10.0
truncation_limits = [5.0, 10.0]

for t_max in truncation_limits:
    for pulse, alpha in product(pulse_list, alpha_values):
        key = f"{pulse}_SNR{snr}_alpha{alpha}_trunc{int(t_max)}"
        base_pulse = _resolve_pulse(pulse)
        kwargs = pulse_kwargs_dict.get(pulse, {})
        resolved_base = partial(base_pulse, **kwargs) if kwargs else base_pulse
        resolved_pulse = truncate_pulse(resolved_base, t_max)

        results[key] = ber_isi_closed_form(
            pulse=resolved_pulse,
            alpha=alpha,
            snr_db=snr,
            offsets=offsets,
            nbits=nbits,
            M=M,
            omega=omega
        )


# Solo para SNR=15 dB, SIR=10 dB y L=2
results_cci = {}
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

        results_cci[key] = ber_cci_closed_form(
            pulse    = resolved_pulse,
            alpha    = alpha,
            snr_db   = snr,
            sir_db   = sir,
            L        = L,
            offsets  = offsets,
            M        = M,
            omega    = omega
        )


results_joint = {}
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

        results_joint[key] = ber_cci_isi_closed_form(
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


print("\n\n% === ISI ===\n")
export_flat_latex_table_truncated(results)

print("\n\n% === CCI ===\n")
export_cci_latex_table_truncated(results_cci)

print("\n\n% === ISI + CCI ===\n")
export_joint_latex_table_truncated(results_joint)
from itertools import product
from functools import partial
from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)

# ─────────────────────────────────────────────────────────────
# 1. Funciones para exportar tablas LaTeX
# ─────────────────────────────────────────────────────────────

def export_flat_latex_table(results, filename=None):
    """Exporta resultados BER (solo ISI) en formato LaTeX plano."""
    def fmt(x): return f"{x:.6e}"
    pulse_rename = {"raised_cosine": "RC", "btrc": "BTRC", "elp": "ELP", "iplcp": "IPLCP"}

    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{BER para diferentes valores de SNR y alpha.}",
        "\\label{tab:ber}",
        "\\begin{tabular}{|l|l|l|r|r|r|r|}",
        "\\hline",
        "snr & alpha & pulse &      0.05 &       0.1 &       0.2 &      0.25 \\\\",
        "\\hline"
    ]

    for key, ber in results.items():
        parts = key.rsplit("_", maxsplit=2)
        pulse, snr_str, alpha_str = parts[0], parts[1][3:], parts[2][5:]
        pulse_label = pulse_rename.get(pulse, pulse.upper())
        row = f"{int(float(snr_str))} & {alpha_str} & {pulse_label} & " + " & ".join(fmt(v) for v in ber) + " \\\\"
        lines.append(row)

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    latex_code = "\n".join(lines)
    print(latex_code) if filename is None else open(filename, "w").write(latex_code)


def export_cci_latex_table(results_cci, filename=None):
    """Exporta resultados BER para CCI en formato LaTeX."""
    def fmt(x): return f"{x:.6e}"
    pulse_rename = {"raised_cosine": "RC", "btrc": "BTRC", "elp": "ELP", "iplcp": "IPLCP"}

    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{BER debido a CCI para diferentes valores de $\\alpha$, SIR y $L$.}",
        "\\label{tab:ber_cci}",
        "\\begin{tabular}{|l|l|l|l|l|r|r|r|r|}",
        "\\hline",
        "SNR & SIR & $\\alpha$ & $L$ & pulse & 0.05 & 0.10 & 0.20 & 0.25 \\\\",
        "\\hline"
    ]

    for key, ber in results_cci.items():
        parts = key.rsplit("_", maxsplit=4)
        pulse, _, sir_str, alpha_str, L_str = parts
        pulse_label = pulse_rename.get(pulse, pulse.upper())
        row = f"15 & {sir_str[3:]} & {alpha_str[5:]} & {L_str[1:]} & {pulse_label} & " + " & ".join(fmt(v) for v in ber) + " \\\\"
        lines.append(row)

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    latex_code = "\n".join(lines)
    print(latex_code) if filename is None else open(filename, "w").write(latex_code)


def export_joint_latex_table(results_joint, filename=None):
    """Exporta resultados BER para ISI + CCI en formato LaTeX."""
    def fmt(x): return f"{x:.6e}"
    pulse_rename = {"raised_cosine": "RC", "btrc": "BTRC", "elp": "ELP", "iplcp": "IPLCP"}

    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{BER debido a ISI + CCI para $L=6$, SNR = SIR = 15 dB, y distintos valores de $\\alpha$.}",
        "\\label{tab:ber_joint}",
        "\\begin{tabular}{|l|l|r|r|r|r|r|}",
        "\\hline",
        "pulse & $\\alpha$ & 0.05 & 0.10 & 0.20 & 0.25 \\\\",
        "\\hline"
    ]

    for key, ber in results_joint.items():
        parts = key.rsplit("_", maxsplit=5)
        pulse, _, _, alpha_str, _, _ = parts
        pulse_label = pulse_rename.get(pulse, pulse.upper())
        row = f"{pulse_label} & {alpha_str[5:]} & " + " & ".join(fmt(v) for v in ber) + " \\\\"
        lines.append(row)

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    latex_code = "\n".join(lines)
    print(latex_code) if filename is None else open(filename, "w").write(latex_code)

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

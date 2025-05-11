from itertools import product
from functools import partial
import numpy as np
import re
from ber_toolbox import (
    _resolve_pulse,
    ber_isi_closed_form,
    ber_cci_closed_form,
    ber_cci_isi_closed_form
)

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


def truncate_pulse(base_pulse, t_max: float):
    """
    Returns a pulse function truncated to [-t_max, t_max].

    Parameters
    ----------
    base_pulse : Callable
        Original pulse function g(t, alpha).
    t_max : float
        Truncation limit in time units (typically t/T).

    Returns
    -------
    Callable
        Truncated pulse function.
    """
    def g_truncated(t, alpha):
        out = base_pulse(t, alpha)
        out[np.abs(t) > t_max] = 0.0
        return out
    return g_truncated


def fmt(x): 
    return f"{x:.6e}"




def export_flat_latex_table_truncated(results, filename=None):
    """
    Exporta BER (ISI) en subtables por SNR dentro de un table* para cada truncamiento.
    """

    pulse_rename = {
        "raised_cosine": "RC",
        "btrc": "BTRC",
        "elp": "ELP",
        "iplcp": "IPLCP"
    }

    # Extraer todos los valores de trunc y snr existentes
    trunc_values = sorted({int(m.group("trunc"))
                           for key in results
                           for m in [re.match(
                               r".+_SNR[\d\.]+_alpha[\d\.]+_trunc(?P<trunc>\d+)", key)]
                           if m})
    snr_values = sorted({float(m.group("snr"))
                         for key in results
                         for m in [re.match(
                             r".+_SNR(?P<snr>[\d\.]+)_alpha[\d\.]+_trunc\d+", key)]
                         if m})

    lines = []
    for trunc in trunc_values:
        lines.append(r"\begin{table*}[h!]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{BER ISI trunc=±{trunc} T para distintos SNR y $\\alpha$.}}")
        lines.append(f"\\label{{tab:ber_isi_trunc{trunc}}}")

        for snr in snr_values:
            lines.append(r"\begin{subtable}[t]{0.48\textwidth}")
            lines.append(r"\centering\scriptsize")
            lines.append(f"\\caption{{SNR = {int(snr)} dB}}")
            lines.append(r"\begin{tabular}{|l|l|r|r|r|r|}")
            lines.append(r"\hline")
            lines.append(r"$\alpha$ & pulse & 0.05 & 0.10 & 0.20 & 0.25 \\")
            lines.append(r"\hline")

            # Filtrar y ordenar por alpha
            entries = []
            pattern = re.compile(
                rf"(?P<pulse>.+)_SNR{snr}_alpha(?P<alpha>[\d\.]+)_trunc{trunc}"
            )
            for key, ber in results.items():
                m = pattern.match(key)
                if not m:
                    continue
                alpha = float(m.group("alpha"))
                p_lbl = pulse_rename.get(m.group("pulse"), m.group("pulse").upper())
                row = f"{alpha:.2f} & {p_lbl} & " + " & ".join(fmt(v) for v in ber) + r" \\"
                entries.append((alpha, row))

            for _, row in sorted(entries, key=lambda x: x[0]):
                lines.append(row)

            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{subtable}%")
            lines.append(r"\hfill")

        lines.append(r"\end{table*}")
        lines.append("")

    latex_code = "\n".join(lines)
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    else:
        print(latex_code)
        
        
        
        
def export_cci_latex_table_truncated(results_cci, filename=None):
    """
    Exporta BER (CCI) en un table por cada truncamiento,
    mostrando sólo pulse y alpha (SNR=15 dB, SIR=10 dB, L=2 constantes).
    """

    pulse_rename = {
        "raised_cosine": "RC",
        "btrc": "BTRC",
        "elp": "ELP",
        "iplcp": "IPLCP"
    }

    # Determinar valores de trunc disponibles en las claves
    trunc_values = sorted({
        int(m.group("trunc"))
        for key in results_cci
        for m in [re.match(
            r".+_SNR15\.0_SIR10\.0_alpha[\d\.]+_L2_trunc(?P<trunc>\d+)",
            key)]
        if m
    })

    lines = []
    for trunc in trunc_values:
        lines.append(r"\begin{table}[h!]")
        lines.append(r"\centering\scriptsize")
        lines.append(f"\\caption{{BER CCI trunc=±{trunc}\\,T para SNR=15\\,dB, SIR=10\\,dB, L=2.}}")
        lines.append(f"\\label{{tab:ber_cci_trunc{trunc}}}")
        lines.append(r"\begin{tabular}{|l|l|r|r|r|r|}")
        lines.append(r"\hline")
        lines.append(r"pulse & $\alpha$ & 0.05 & 0.10 & 0.20 & 0.25 \\")
        lines.append(r"\hline")

        # Patron para extraer pulse y alpha
        pattern = re.compile(
            rf"(?P<pulse>.+)_SNR15\.0_SIR10\.0_alpha(?P<alpha>[\d\.]+)_L2_trunc{trunc}"
        )
        entries = []
        for key, ber in results_cci.items():
            m = pattern.match(key)
            if not m:
                continue
            alpha = float(m.group("alpha"))
            p_lbl = pulse_rename.get(m.group("pulse"), m.group("pulse").upper())
            row = (
                f"{p_lbl} & {alpha:.2f} & " +
                " & ".join(fmt(v) for v in ber) +
                r" \\"
            )
            entries.append((alpha, row))

        # Ordenar filas por alpha
        for _, row in sorted(entries, key=lambda x: x[0]):
            lines.append(row)

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    latex_code = "\n".join(lines)
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    else:
        print(latex_code)


def export_joint_latex_table_truncated(results_joint, filename=None):
    """
    Exporta BER (ISI+CCI) en un table por truncamiento, filas ordenadas por α.
    """

    pulse_rename = {
        "raised_cosine": "RC",
        "btrc": "BTRC",
        "elp": "ELP",
        "iplcp": "IPLCP"
    }

    trunc_values = sorted({int(m.group("trunc"))
                           for key in results_joint
                           for m in [re.match(
                               r".+_SNR15\.0_SIR15\.0_alpha[\d\.]+_L6_joint_trunc(?P<trunc>\d+)",
                               key)]
                           if m})

    lines = []
    for trunc in trunc_values:
        lines.append(r"\begin{table}[h!]")
        lines.append(r"\centering\scriptsize")
        lines.append(f"\\caption{{BER ISI+CCI trunc=±{trunc} T para SNR=SIR=15 dB, $L=6$.}}")
        lines.append(f"\\label{{tab:ber_joint_trunc{trunc}}}")
        lines.append(r"\begin{tabular}{|l|l|r|r|r|r|}")
        lines.append(r"\hline")
        lines.append(r"pulse & $\alpha$ & 0.05 & 0.10 & 0.20 & 0.25 \\")
        lines.append(r"\hline")

        pattern = re.compile(
            rf"(?P<pulse>.+)_SNR15\.0_SIR15\.0_alpha(?P<alpha>[\d\.]+)_L6_joint_trunc{trunc}"
        )
        entries = []
        for key, ber in results_joint.items():
            m = pattern.match(key)
            if not m:
                continue
            alpha = float(m.group("alpha"))
            p_lbl = pulse_rename.get(m.group("pulse"), m.group("pulse").upper())
            row = f"{p_lbl} & {alpha:.2f} & " + " & ".join(fmt(v) for v in ber) + r" \\"
            entries.append((alpha, row))

        for _, row in sorted(entries, key=lambda x: x[0]):
            lines.append(row)

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    latex_code = "\n".join(lines)
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    else:
        print(latex_code)






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

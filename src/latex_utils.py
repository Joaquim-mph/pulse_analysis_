import re
import numpy as np

pulse_rename = {
    "raised_cosine": "RC",
    "btrc":         "BTRC",
    "elp":          "ELP",
    "iplcp":        "IPLCP",
}



def fmt(x):
    return f"{x:.2e}"


def truncate_pulse(base_pulse, t_max):
    def g_trunc(t, alpha):
        out = base_pulse(t, alpha)
        return out * (np.abs(t) <= t_max)
    return g_trunc


def _parse_key(key, pattern):
    m = re.match(pattern, key)
    return m.groupdict() if m else None


def export_flat_latex_table(results, filename=None):
    """Exporta resultados BER (solo ISI) en formato LaTeX plano."""

    # Recolectar y parsear las tuplas (snr, alpha, pulse_label, ber)
    entries = []
    for key, ber in results.items():
        # key con formato "pulse_SNR{snr}_alpha{alpha}"
        parts = key.rsplit("_", maxsplit=2)
        pulse = parts[0]
        snr = float(parts[1][3:])
        alpha = float(parts[2][5:])
        pulse_label = pulse_rename.get(pulse, pulse.upper())
        entries.append((snr, alpha, pulse_label, ber))

    # Ordenar primero por snr ascendente, luego por alpha ascendente
    entries.sort(key=lambda x: (x[0], x[1]))

    # Encabezado de la tabla
    lines = [
        "\\begin{table}[h!]",
        "\\centering\\scriptsize",
        "\\caption{BER para diferentes valores de SNR y $\\alpha$.}",
        "\\label{tab:ber}",
        "\\begin{tabular}{|l|l|l|r|r|r|r|}",
        "\\hline",
        "snr & $\\alpha$ & pulse & 0.05 & 0.10 & 0.20 & 0.25 \\\\",
        "\\hline"
    ]

    # Rellenar filas en el orden deseado
    for snr, alpha, pulse_label, ber in entries:
        row = (
            f"{int(snr)} & {alpha:.2f} & {pulse_label} & "
            + " & ".join(fmt(v) for v in ber)
            + " \\\\"
        )
        lines.append(row)

    # Pie de tabla
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]

    latex_code = "\n".join(lines)
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    else:
        print(latex_code)
        

def export_cci_latex_table(results_cci, filename=None):
    """
    Exporta BER (CCI) en dos tablas, una por cada valor de SIR,
    asumiendo SNR=15 dB constante (omitido de las columnas).
    Cada tabla muestra columnas: alpha, L, pulse y offsets.
    """
    pulse_rename = {
        "raised_cosine": "RC",
        "btrc": "BTRC",
        "elp": "ELP",
        "iplcp": "IPLCP"
    }

    # Extraer valores únicos de SIR de las claves
    sir_values = sorted({
        float(m.group("sir"))
        for key in results_cci
        for m in [re.match(
            r".+_SNR15\.0_SIR(?P<sir>[\d\.]+)_alpha[\d\.]+_L\d+", key)]
        if m
    })

    tables = []
    for sir in sir_values:
        lines = [
            r"\begin{table}[h!]",
            r"\centering\scriptsize",
            f"\\caption{{BER CCI para SIR={int(sir)}\\,dB, distintos $\\alpha$ y $L$ (SNR fijo 15\\,dB).}}",
            f"\\label{{tab:ber_cci_sir{int(sir)}}}",
            r"\begin{tabular}{|l|l|l|r|r|r|r|}",
            r"\hline",
            r"$\alpha$ & $L$ & pulse & 0.05 & 0.10 & 0.20 & 0.25 \\",
            r"\hline"
        ]

        # Patrón que captura pulse, alpha y L (SNR y SIR se asumen)
        pattern = re.compile(
            rf"(?P<pulse>.+)_SNR15\.0_SIR{sir}_alpha(?P<alpha>[\d\.]+)_L(?P<L>\d+)"
        )
        entries = []
        for key, ber in results_cci.items():
            m = pattern.match(key)
            if not m:
                continue
            alpha = float(m.group("alpha"))
            L = int(m.group("L"))
            p_lbl = pulse_rename.get(m.group("pulse"), m.group("pulse").upper())
            row = (
                f"{alpha:.2f} & {L} & {p_lbl} & " +
                " & ".join(fmt(v) for v in ber) +
                r" \\"
            )
            entries.append((alpha, L, row))

        # Ordenar por alpha, luego L
        for _, _, row in sorted(entries, key=lambda x: (x[0], x[1])):
            lines.append(row)

        lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
        tables.append("\n".join(lines))

    output = "\n".join(tables)
    if filename:
        with open(filename, "w") as f:
            f.write(output)
    else:
        print(output)



def export_joint_latex_table(results_joint, filename=None):
    """Exporta resultados BER para ISI + CCI en formato LaTeX."""

    pulse_rename = {"raised_cosine": "RC", "btrc": "BTRC", "elp": "ELP", "iplcp": "IPLCP"}

    lines = [
        "\\begin{table}[h!]",
        "\\centering\scriptsize",
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


        
        

def export_flat_latex_table_truncated(results, filename=None):
    """
    Exporta BER (ISI) creando un table* independiente para cada
    combinación de truncamiento y SNR.
    """
    pulse_rename = {
        "raised_cosine": "RC",
        "btrc": "BTRC",
        "elp": "ELP",
        "iplcp": "IPLCP"
    }


    # Extraer todos los valores de trunc y snr existentes
    trunc_values = sorted({
        int(m.group("trunc"))
        for key in results
        for m in [re.match(
            r".+_SNR[\d\.]+_alpha[\d\.]+_trunc(?P<trunc>\d+)", key)]
        if m
    })
    snr_values = sorted({
        float(m.group("snr"))
        for key in results
        for m in [re.match(
            r".+_SNR(?P<snr>[\d\.]+)_alpha[\d\.]+_trunc\d+", key)]
        if m
    })

    lines = []

    for trunc in trunc_values:
        for snr in snr_values:
            lines.append(r"\begin{table*}[h!]")
            lines.append(r"\centering\scriptsize")
            lines.append(
                f"\\caption{{BER ISI trunc=±{trunc}\\,T, SNR={int(snr)}\\,dB y distintos $\\alpha$.}}"
            )
            lines.append(f"\\label{{tab:ber_isi_trunc{trunc}_snr{int(snr)}}}")
            lines.append(r"\begin{tabular}{|l|l|r|r|r|r|}")
            lines.append(r"\hline")
            lines.append(r"$\alpha$ & pulse & 0.05 & 0.10 & 0.20 & 0.25 \\")
            lines.append(r"\hline")

            # Filtrar y ordenar por alpha
            pattern = re.compile(
                rf"(?P<pulse>.+)_SNR{snr}_alpha(?P<alpha>[\d\.]+)_trunc{trunc}"
            )
            entries = []
            for key, ber in results.items():
                m = pattern.match(key)
                if not m:
                    continue
                alpha = float(m.group("alpha"))
                p_lbl = pulse_rename.get(m.group("pulse"), m.group("pulse").upper())
                row = (
                    f"{alpha:.2f} & {p_lbl} & "
                    + " & ".join(fmt(v) for v in ber)
                    + r" \\"
                )
                entries.append((alpha, row))

            for _, row in sorted(entries, key=lambda x: x[0]):
                lines.append(row)

            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table*}")
            lines.append("")  # blank line between floats

    latex_code = "\n".join(lines)
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    else:
        print(latex_code)


        

def export_cci_latex_table_truncated(results_cci, filename=None):
    """
    Exporta BER (CCI) en un table por cada truncamiento,
    mostrando sólo pulse y alpha (SNR=15 dB, SIR=10 dB, L=2 constantes).
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
        lines.append(f"\\caption{{BER ISI+CCI trunc=±{trunc} T para SNR=SIR=15 dB, $L=6$.}}")
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





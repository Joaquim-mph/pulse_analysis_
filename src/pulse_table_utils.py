# pulse_table_utils.py

import re
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

# ─── 1. CONSTANTS & PRECOMPILED REGEXES ────────────────────────────────────────

# single source of truth for how we abbreviate pulses
PULSE_RENAME: dict[str,str] = {
    "raised_cosine": r"\bfseries RC",
    "btrc":          r"\bfseries BTRC",
    "elp":           r"\bfseries ELP",
    "iplcp":         r"\bfseries IPLCP",
}

# generic parser regex: captures
#   pulse, snr, optional sir, alpha, optional L, optional trunc, optional 'joint' flag
_RE_GENERIC = re.compile(
    r"^"
    r"(?P<pulse>.+?)"
    r"_SNR(?P<snr>[\d\.]+)"
    r"(?:_SIR(?P<sir>[\d\.]+))?"
    r"_alpha(?P<alpha>[\d\.]+)"
    r"(?:_L(?P<L>\d+))?"
    r"(?:_joint)?"                     
    r"(?:_trunc(?P<trunc>\d+))?"       
    r"$"
)



def truncate_pulse(base_pulse, t_max):
    def g_trunc(t, alpha):
        out = base_pulse(t, alpha)
        return out * (np.abs(t) <= t_max)
    return g_trunc


def results_to_df(results: dict) -> pd.DataFrame:
    """
    Parse a results dict into a unified DataFrame.

    Each key in `results` should match _RE_GENERIC, and the value
    should be an iterable of four BER floats corresponding to offsets
    [0.05, 0.10, 0.20, 0.25].

    Returns a DataFrame with columns:
      - pulse: abbreviated pulse name (e.g. "RC", "BTRC")
      - snr:   float SNR value
      - sir:   float SIR value or None
      - alpha: float roll-off
      - L:     int number of interferers or None
      - trunc: int truncation limit or None
      - joint: bool flag for joint ISI+CCI
      - ber05, ber10, ber20, ber25: float BER values at each offset
    """
    rows = []
    for key, ber in results.items():
        m = _RE_GENERIC.match(key)
        if not m:
            continue
        gd = m.groupdict()
        pulse_label = PULSE_RENAME.get(gd["pulse"], gd["pulse"].upper())
        snr   = float(gd["snr"])
        sir   = float(gd["sir"])   if gd["sir"]   else None
        alpha = float(gd["alpha"])
        L     = int(gd["L"])       if gd["L"]     else None
        trunc = int(gd["trunc"])   if gd["trunc"] else None
        joint = bool(key.endswith("_joint"))
        
        # Expand the BER array into separate columns
        ber05, ber10, ber20, ber25 = list(ber)
        
        rows.append({
            "pulse": pulse_label,
            "snr": snr,
            "sir": sir,
            "alpha": alpha,
            "L": L,
            "trunc": trunc,
            "joint": joint,
            "ber05": ber05,
            "ber10": ber10,
            "ber20": ber20,
            "ber25": ber25,
        })
    
    return pd.DataFrame(rows)






def latex_table(
    df: pd.DataFrame,
    caption: str = None,
    label: str = None,
    filename: str = None,
    float_format: str = "%.2e"
) -> str:
    """
    Automatically generates a LaTeX table from `df`, dropping any columns
    that are entirely None/NaN *or* that are constant, aligning numeric
    columns to the right, and ordering rows by snr, then sir (if present),
    then alpha, then L.
    """
    # 1) Replace None with NA, convert dtypes, and drop all‐NA columns
    df2 = df.replace({None: pd.NA}).convert_dtypes()
    df2 = df2.dropna(axis=1, how='all')

    # 2) Drop any *constant* columns (only one unique value after dropping NAs)
    df2 = df2.loc[:, df2.nunique(dropna=True) > 1]

    # 3) Sort rows by whichever of snr, sir, alpha, L remain
    sort_order = []
    for col in ["snr", "sir", "alpha", "L"]:
        if col in df2.columns:
            sort_order.append(col)
    if sort_order:
        df2 = df2.sort_values(by=sort_order)

    # 4) Prepare a LaTeX‐style header, but DO NOT rename df2.columns
    display_names = {
        "pulse": r"\bfseries Pulse",
        "snr":   r"\bfseries SNR (dB)",
        "sir":   r"\bfseries SIR (dB)",
        "alpha": r"$\alpha$",
        "L":     r"$L$",
        "trunc": r"\bfseries trunc",
        "ber05": r"$t/T=0.05$",
        "ber10": r"$t/T=0.10$",
        "ber20": r"$t/T=0.20$",
        "ber25": r"$t/T=0.25$",
    }
    header = [display_names.get(col, col) for col in df2.columns]

    # 5) Alignment, formatters, etc. stay the same
    aligns = [ "c" if pd.api.types.is_numeric_dtype(df2[c])
               else "l"
               for c in df2.columns ]
    column_format = "|" + "|".join(aligns) + "|"
    formatters = {"alpha": lambda x: f"{x:.2f}"}

    # 6) Pass that header list in; do not rename df2 itself
    latex = df2.to_latex(
        index=False,
        header=header,
        float_format=float_format,
        formatters=formatters,
        column_format=column_format,
        caption=caption,
        label=label,
        escape=False
    )

    # 7) Emit or write to file
    if filename:
        Path(filename).write_text(latex)
        print(f"Wrote LaTeX table to {filename}")
    else:
        print(latex)

    return latex
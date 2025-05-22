# pulse_table_utils.py

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ─── 1. CONSTANTS & PRECOMPILED REGEXES ────────────────────────────────────────

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

    Each key in results should match _RE_GENERIC, and the value
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
        
        ber05, ber10, ber20, ber25 = list(ber)
        
        rows.append({
            "pulse": pulse_label,
            "snr":   snr,
            "sir":   sir,
            "alpha": alpha,
            "L":     L,
            "trunc": trunc,
            "joint": joint,
            "ber05": ber05,
            "ber10": ber10,
            "ber20": ber20,
            "ber25": ber25,
        })
    
    df = pd.DataFrame(rows)

    # ———  NEW: apply formatting ——————————————————————————————
    # a dict mapping column → formatter(x)→ str
    formatters = {
        "alpha": lambda x: f"{x:.2f}",
        # example for BER columns in scientific notation:
        "ber05": lambda x: f"{x:.2e}",
        "ber10": lambda x: f"{x:.2e}",
        "ber20": lambda x: f"{x:.2e}",
        "ber25": lambda x: f"{x:.2e}",
        # you can add "snr", "sir", etc. here too if you wish
    }

    for col, fmt in formatters.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: fmt(v) if pd.notna(v) else ""
            )

    return df



def save_df_to_csv(
    df: pd.DataFrame,
    filename: str,
    folder: str = "results"
) -> None:
    """
    Save a DataFrame to CSV inside `folder`, dropping columns that are entirely
    None/NaN and turning any remaining None/NaN into blank cells. Creates the
    folder if it doesn't exist. Columns in `snr`, `sir`, `alpha`, `L` will be
    moved to the front if they exist. Rows will be sorted by these columns
    if present.
    """
    # Ensure the output folder exists
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop fully-empty columns and blank out remaining NAs
    df_clean = df.dropna(axis=1, how="all").fillna("")

    # Reorder columns: bring ['snr','sir','alpha','L'] to front, if present
    priority = ["snr", "sir", "alpha", "L"]

    # Sort rows by whichever of snr, sir, alpha, L remain
    sort_order = [c for c in priority if c in df_clean.columns]
    if sort_order:
        df_clean = df_clean.sort_values(by=sort_order)

    # Write to CSV in the specified folder
    csv_path = out_dir / filename
    df_clean.to_csv(csv_path, index=False)



def latex_table(
    df: pd.DataFrame,
    caption: str = None,
    label: str = None,
    filename: str = None,
    float_format: str = "%.2e"
) -> str:
    """
    Generates a LaTeX table using booktabs + siunitx + multirow.
    Numeric columns become S-columns (decimal aligned, sci-notation).
    Text columns (pulse, etc.) become left-aligned l-columns.
    """
    # 1) Drop all-NA and constant cols
    df2 = df.replace({None: pd.NA}).convert_dtypes()
    df2 = df2.dropna(axis=1, how='all')
    df2 = df2.loc[:, df2.nunique(dropna=True) > 1]

    # 2) Sort by grouping keys
    sort_cols = [c for c in ("snr", "sir", "trunc", "L", "alpha") if c in df2.columns]
    if sort_cols:
        df2 = df2.sort_values(by=sort_cols).reset_index(drop=True)

    # 3) Remember which were numeric, then format them per-column
    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]

    offset_cols = [c for c in numeric_cols if c.startswith("ber")]
    int_cols    = [c for c in numeric_cols if c in ("snr","sir","L","trunc")]
    alpha_cols  = [c for c in numeric_cols if c == "alpha"]

    for c in offset_cols:
        df2[c] = df2[c].map(lambda x: float_format % x if pd.notna(x) else "")

    for c in int_cols:
        df2[c] = df2[c].map(lambda x: str(int(x))     if pd.notna(x) else "")

    for c in alpha_cols:
        df2[c] = df2[c].map(lambda x: f"{x:.2f}"       if pd.notna(x) else "")

    # 4) Make a string copy and collapse repeats into \multirow
    df3 = df2.astype(str).copy()
    for col in sort_cols:
        vals, newcol = df3[col], []
        idx = 0
        while idx < len(vals):
            val = vals.iat[idx]
            run = 1
            while idx+run < len(vals) and vals.iat[idx+run] == val:
                run += 1
            if run > 1:
                newcol.append(f"\\multirow{{{run}}}{{*}}{{{val}}}")
                newcol.extend([""]*(run-1))
            else:
                newcol.append(val)
            idx += run
        df3[col] = newcol

    # 5) Build header and col spec (no '|'s!)
    display_names = {
        "pulse": r"\bfseries Pulse",
        "snr":   r"\bfseries SNR (dB)",
        "sir":   r"\bfseries SIR (dB)",
        "alpha": r"$\alpha$",
        "L":     r"$L$",
        "trunc": r"\bfseries trunc",
        "ber05": r"$t/T= \pm 0.05$",
        "ber10": r"$t/T= \pm 0.10$",
        "ber20": r"$t/T= \pm 0.20$",
        "ber25": r"$t/T= \pm 0.25$",
        "joint": r"\bfseries joint"
    }
    cols   = list(df3.columns)
    header = " & ".join(display_names.get(c, c) for c in cols) + r" \\"

    # each numeric col → S, text col → l
    aligns = ["c" if c in numeric_cols else "l" for c in cols]
    col_spec = "".join(aligns)
    ncol     = len(cols)

    # 6) Body with \addlinespace + \cmidrule between groups
    lines = []
    if sort_cols:
        for _, idx_list in df2.groupby(sort_cols).indices.items():
            for idx in idx_list:
                row = df3.iloc[idx].tolist()
                lines.append("  " + " & ".join(row) + r" \\")
            lines.append("  \\addlinespace")
            lines.append(f"  \\cmidrule{{1-{ncol}}}")
    else:
        for _, row in df3.iterrows():
            lines.append("  " + " & ".join(row.tolist()) + r" \\")
    body = "\n".join(lines) + "\n"

    # 7) Wrap in table environment
    tbl  = r"\begin{table}[h!]" + "\n"
    if caption:
        tbl += f"  \\caption{{{caption}}}\n"
    if label:
        tbl += f"  \\label{{{label}}}\n"
    tbl += "  \\centering\n"
    tbl += f"  \\begin{{tabular}}{{{col_spec}}}\n"
    tbl += "    \\toprule\n"
    tbl += "    " + header + "\n"
    tbl += "    \\midrule\n"
    tbl += body
    tbl += "    \\bottomrule\n"
    tbl += "  \\end{tabular}\n"
    tbl += r"\end{table}" + "\n"

    # 8) Output
    if filename:
        Path(filename).write_text(tbl)
        print(f"Wrote LaTeX table to {filename}")
    else:
        print(tbl)
    return tbl
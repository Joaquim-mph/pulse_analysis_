# pulse_table_utils.py

import re
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

# ─── 1. CONSTANTS & PRECOMPILED REGEXES ────────────────────────────────────────

# single source of truth for how we abbreviate pulses
PULSE_RENAME: dict[str,str] = {
    "raised_cosine": "RC",
    "btrc":          "BTRC",
    "elp":           "ELP",
    "iplcp":         "IPLCP",
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
    r"(?:_trunc(?P<trunc>\d+))?"
    r"(?:_joint)?"
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




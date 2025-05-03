# ber_toolbox.py ──────────────────────────────────────────────────────
"""
Closed‑form BER for a BPSK system with inter‑symbol interference (ISI)
caused by an arbitrary pulse shape g(t).

It implements Craig’s series expression:

    Pe = ½ – (2/π) Σ_{m odd}  e^{-(mω)^2/2} sin(mωg0) / m ⋅ Π_k cos(mωgk)

where
  • g0 = coeff · g(τ)             main‑tap contribution
  • gk = coeff · a_k g(τ – k)     ISI taps (a_k ∈ {±1} for BPSK)
  • ω  = noise std (in radians)   (0.10 used in the legacy code)

The function returns Pe for the four timing offsets
τ/T ∊ {0.05, 0.10, 0.20, 0.25}.
"""

from typing import Callable, Sequence, Union, Optional
import importlib
import numpy as np
from numpy.typing import NDArray
from scipy.special import j0  


def _resolve_pulse(
    pulse: Union[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
    """
    Accepts a pulse function or its string key (looked up in pulse_toolbox).
    Returns the callable g(t, alpha).
    """
    if isinstance(pulse, str):
        pulse = getattr(importlib.import_module("pulse_toolbox"), pulse)
    return pulse



def ber_isi_closed_form(
    pulse: Union[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]],
    alpha: float,
    snr_db: float,
    *,
    nbits: int = 1024,
    M: int = 100,
    omega: float = 0.10,
    offsets: Sequence[float] = (0.05, 0.10, 0.20, 0.25),
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Closed‑form BER for **ISI only** (Craig, Eq. 7).
    """
    g = _resolve_pulse(pulse)

    if rng is None:
        rng = np.random.default_rng()

    T      = 1.0
    N      = nbits // 2                 # ISI span ±N symbols
    coeff  = 10 ** (snr_db / 20)        # desired‑signal amplitude

    ab        = np.concatenate((np.arange(-N, 0), np.arange(1, N + 1)))
    m         = np.arange(1, M, 2, dtype=float)   # odd terms
    m_omega   = m * omega
    exp_term  = np.exp(-(m_omega**2)/2) / m

    ber = np.empty(len(offsets), dtype=float)

    for i, tau in enumerate(offsets):
        # all taps in one vectorised pulse call
        t_vals = np.concatenate( ((tau,), tau - ab) )
        g_vals = g(t_vals * T, alpha)

        g0 = coeff * g_vals[0]                                     # main tap
        gk = coeff * rng.choice((-1.0, 1.0), size=ab.size) * g_vals[1:]  # ISI taps

        prod_cos = np.prod(np.cos(m_omega[:, None] * gk[None, :]), axis=1)
        suma     = np.sum(exp_term * np.sin(m_omega * g0) * prod_cos)
        ber[i]   = 0.5 - (2 / np.pi) * suma

    return ber




def ber_cci_closed_form(
    pulse: Union[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]],
    alpha: float,
    snr_db: float,
    sir_db: float,
    *,
    L: int = 2,
    M: int = 100,
    omega: float = 0.10,
    offsets: Sequence[float] = (0.05, 0.10, 0.20, 0.25),
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Closed‑form BER due to **co‑channel interference only** .

    Each interfering signal is assumed BPSK with identical pulse shape and
    independent random symbols/phases.  The effective interference tap
    amplitude is scaled so that SIR (in dB) is satisfied **per interferer**.

    Parameters
    ----------
    sir_db : float
        Desired‑to‑interferer power ratio in dB (per interferer).
    L : int
        Number of equal‑power interferers.

    Notes
    -----
    - Interference taps r_i are modelled as ± a with
      ``a = 10**(-sir_db/20)`` and random sign.
    """
    g = _resolve_pulse(pulse)
    if rng is None:
        rng = np.random.default_rng()

    coeff = 10 ** (snr_db / 20)          # desired‑signal amplitude
    a_int = 10 ** (-sir_db / 20)         # interferer amplitude  (per interferer)

    m = np.arange(1, M, 2, dtype=float)          # odd terms
    m_omega = m * omega
    exp_term = np.exp(-(m_omega**2) / 2) / m     # common prefactor

    ber = np.empty(len(offsets))
    for i, tau in enumerate(offsets):
        g0 = coeff * g(np.array([tau]), alpha)[0]
        r_i = a_int * rng.choice((-1.0, 1.0), size=L)          # random CCI taps
        bessel_prod = np.prod(j0(m_omega[:, None] * r_i[None, :]), axis=1)

        suma = np.sum(exp_term * np.sin(m_omega * g0) * bessel_prod)
        ber[i] = 0.5 - (2 / np.pi) * suma
    return ber


def ber_cci_isi_closed_form(
    pulse: Union[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]],
    alpha: float,
    snr_db: float,
    sir_db: float,
    *,
    L: int = 2,
    nbits: int = 1024,
    M: int = 100,
    omega: float = 0.10,
    offsets: Sequence[float] = (0.05, 0.10, 0.20, 0.25),
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Closed‑form BER with **simultaneous ISI and CCI** (Eq. 8).
    """
    g = _resolve_pulse(pulse)
    if rng is None:
        rng = np.random.default_rng()

    T = 1.0
    N = nbits // 2                       # ISI span ±N symbols
    coeff   = 10 ** (snr_db / 20)
    a_int   = 10 ** (-sir_db / 20)

    ab       = np.concatenate((np.arange(-N, 0), np.arange(1, N + 1)))
    m        = np.arange(1, M, 2, dtype=float)
    m_omega  = m * omega
    exp_term = np.exp(-(m_omega**2) / 2) / m

    ber = np.empty(len(offsets))
    for i, tau in enumerate(offsets):
        # Desired‑signal taps
        t_vals = np.concatenate(((tau,), tau - ab))
        g_vals = g(t_vals * T, alpha)
        g0 = coeff * g_vals[0]
        gk = coeff * rng.choice((-1.0, 1.0), size=ab.size) * g_vals[1:]

        # Interference taps
        r_i = a_int * rng.choice((-1.0, 1.0), size=L)

        prod_cos   = np.prod(np.cos(m_omega[:, None] * gk[None, :]), axis=1)
        prod_bessel = np.prod(j0(m_omega[:, None] * r_i[None, :]), axis=1)

        suma = np.sum(exp_term * np.sin(m_omega * g0) * prod_cos * prod_bessel)
        ber[i] = 0.5 - (2 / np.pi) * suma
    return ber

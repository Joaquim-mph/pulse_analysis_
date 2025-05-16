"""
Closed-form BER Computation for BPSK with ISI and CCI
------------------------------------------------------

This module provides closed-form expressions for the bit error rate (BER) in
a BPSK system subject to:

1. Intersymbol interference (ISI),
2. Co-channel interference (CCI), or
3. Simultaneous ISI and CCI.

All expressions follow the analytical framework proposed by
N. C. Beaulieu (2007).

─────────────────────────────────────────────────────────────────────────────

1. ISI-Only Case — `ber_isi_closed_form`
----------------------------------------
Implements Craig's series expression for a BPSK system with ISI:

    Pe = ½ - (2/π) Σ_{m odd} e^{-(mω)²/2} · sin(mω·g₀) / m · Π_k cos(mω·g_k)

where:
  • g₀ = coeff · g(τ)           → main-tap contribution
  • g_k = coeff · a_k·g(τ-k)    → ISI taps (a_k ∈ {±1})
  • ω  = noise std (default 0.10)

─────────────────────────────────────────────────────────────────────────────

2. CCI-Only Case — `ber_cci_closed_form`
----------------------------------------
Implements Beaulieu's approximation for BPSK with L co-channel interferers:

    Pe = ½ - (2/π) Σ_{m odd} e^{-(mω)²/2} · sin(mω·g₀) / m · Π_i J₀(mω·r_i)

where:
  • g₀ = coeff · g(τ)              → desired main tap
  • r_i = ±a (random sign)         → L interferer taps
  • J₀() = Bessel function of the first kind

─────────────────────────────────────────────────────────────────────────────

3. ISI + CCI Case — `ber_cci_isi_closed_form`
---------------------------------------------
Combines Craig's ISI and Beaulieu's CCI approximations:

    Pe = ½ - (2/π) Σ_{m odd} e^{-(mω)²/2} · sin(mω·g₀) / m
                          · Π_k cos(mω·g_k) · Π_i J₀(mω·r_i)

─────────────────────────────────────────────────────────────────────────────

In all cases:
  • g(t, α) is the pulse shape used.
  • τ/T ∊ {0.05, 0.10, 0.20, 0.25} by default (timing offsets).
  • `snr_db`, `sir_db`, and other parameters control noise and interference.
"""

from typing import Callable, Sequence, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.special import j0  
from pulse_toolbox import PULSE_FNS


def _resolve_pulse(
    pulse: Union[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
    """
    Resolves a pulse reference to a callable function.
    """
    if isinstance(pulse, str):
        if pulse not in PULSE_FNS:
            raise ValueError(f"Unknown pulse name '{pulse}' in PULSE_FNS.")
        return PULSE_FNS[pulse]
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
    Computes the closed-form BER due to inter-symbol interference (ISI)
    using Craig's series formula.

    Parameters
    ----------
    pulse : str or callable
        Pulse function to evaluate.
    alpha : float
        Roll-off or shaping factor.
    snr_db : float
        Signal-to-noise ratio in dB.
    nbits : int, optional
        Number of interfering symbols (default is 1024).
    M : int, optional
        Number of odd-series terms to include in the expansion (default is 100).
    omega : float, optional
        Noise angular standard deviation (default is 0.10).
    offsets : sequence of float, optional
        Timing offsets τ/T at which to compute BER.
    rng : numpy.random.Generator, optional
        RNG for generating ISI taps.

    Returns
    -------
    ber : NDArray
        Bit error rate values for each specified timing offset.
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
    Computes the closed-form BER due to co-channel interference (CCI)
    using the expression derived by Beaulieu.

    Parameters
    ----------
    pulse : str or callable
        Pulse function to evaluate.
    alpha : float
        Roll-off or shaping factor.
    snr_db : float
        Signal-to-noise ratio in dB for the desired user.
    sir_db : float
        Signal-to-interferer ratio in dB per interferer.
    L : int, optional
        Number of co-channel interferers (default is 2).
    M : int, optional
        Number of odd-series terms to include in the expansion (default is 100).
    omega : float, optional
        Noise angular standard deviation (default is 0.10).
    offsets : sequence of float, optional
        Timing offsets τ/T at which to compute BER.
    rng : numpy.random.Generator, optional
        RNG for generating CCI taps.

    Returns
    -------
    ber : NDArray
        Bit error rate values for each specified timing offset.
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
    Computes the closed-form BER due to both ISI and CCI,
    combining the expressions from Craig's method and Beaulieu.

    Parameters
    ----------
    pulse : str or callable
        Pulse function to evaluate.
    alpha : float
        Roll-off or shaping factor.
    snr_db : float
        Signal-to-noise ratio in dB.
    sir_db : float
        Signal-to-interferer ratio in dB.
    L : int, optional
        Number of co-channel interferers (default is 2).
    nbits : int, optional
        Number of interfering symbols for ISI computation (default is 1024).
    M : int, optional
        Number of odd-series terms to include in the expansion (default is 100).
    omega : float, optional
        Noise angular standard deviation (default is 0.10).
    offsets : sequence of float, optional
        Timing offsets τ/T at which to compute BER.
    rng : numpy.random.Generator, optional
        RNG for generating both ISI and CCI taps.

    Returns
    -------
    ber : NDArray
        Bit error rate values for each specified timing offset.
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

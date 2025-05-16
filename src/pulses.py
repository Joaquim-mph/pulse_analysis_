import numpy as np
from typing import Callable, Dict


PULSE_FNS: Dict[str, Callable[..., np.ndarray]] = {}

def register(name: str):
    def decorator(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        PULSE_FNS[name] = fn
        return fn
    return decorator


@register('raised_cosine')
def raised_cosine(t: np.ndarray, alpha: float = 0.35, T: float = 1.0) -> np.ndarray:
    """
    p_RC(t) = sinc(t/T) * cos(2παt/T) / (1 - 4α² (t/T)²).
    """
    t = np.asarray(t, dtype=float)
    sinc_part = np.sinc(t / T)
    cos_den = 1 - (2 * alpha * t / T) ** 2
    cos_part = np.where(np.isclose(cos_den, 0.0), np.pi / 4, np.cos(np.pi * alpha * t / T) / cos_den)
    return sinc_part * cos_part



@register('btrc')
def btrc_pulse(t: np.ndarray, alpha: float = 0.35, T: float = 1.0) -> np.ndarray:
    """
    BTRC pulse as defined in the referenced paper.
    """
    t = np.asarray(t, dtype=float)
    pi = np.pi
    beta = (2 * T * np.log(2)) / alpha

    sinc_term = np.sinc(t / T)
    num = (
        4 * beta * pi * t * np.sin(pi * alpha * t / T)
        + 2 * beta**2 * np.cos(pi * alpha * t / T)
        - beta**2
    )
    denom = 4 * pi**2 * t**2 + beta**2

    return sinc_term * (num / denom)


@register('elp')
def elp_pulse(t: np.ndarray, alpha: float = 0.35, beta: float = 0.1, T: float = 1.0) -> np.ndarray:
    tT = t / T
    return np.exp(-np.pi * beta / 2 * tT**2) * np.sinc(tT) * np.sinc(alpha * tT)


@register('iplcp')
def iplcp_pulse(t: np.ndarray, alpha: float = 0.35, mu: float = 1.6, gamma: float = 1.0, epsilon: float = 0.1, T: float = 1.0) -> np.ndarray:
    """
    Time-domain IPLCP pulse, as defined in the equation using normalized time τ = t/T.

    Parameters
    ----------
    t : np.ndarray
        Time axis (in seconds).
    alpha : float
        Roll-off factor.
    mu : float
        Linear shaping parameter.
    gamma : float
        Exponent for spectral shaping.
    epsilon : float
        Gaussian envelope control.
    T : float
        Symbol period.
    """
    t = np.asarray(t, dtype=float)
    tau = t / T

    pi_tau = np.pi * tau
    pi_alpha_tau = np.pi * alpha * tau
    denom = (np.pi ** 2) * (alpha ** 2) * (tau ** 2)

    # Avoid division by zero (handled after for tau=0)
    denom_safe = np.where(denom == 0, 1e-12, denom)

    sinc_tau = np.sinc(tau)
    term1 = 4 * (1 - mu) * (np.sin(pi_alpha_tau / 2) ** 2)
    term2 = np.pi * alpha * mu * tau * np.sin(pi_alpha_tau)

    bracket = (term1 + term2) / denom_safe
    envelope = np.exp(-epsilon * (np.pi ** 2) * (tau ** 2))
    pulse = envelope * (sinc_tau * bracket) ** gamma

    # Fix singularity at tau = 0 explicitly
    center_idx = np.argmin(np.abs(tau))
    pulse[center_idx] = envelope[center_idx] * 1.0

    return pulse


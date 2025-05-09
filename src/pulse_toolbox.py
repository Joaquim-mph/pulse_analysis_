import numpy as np
from typing import Tuple, Optional, Callable, Dict, Union


def t_axis(
    span_T: float,
    oversample: int = 8,
    T: float = 1.0
) -> Tuple[np.ndarray, float]:
    if not isinstance(oversample, int) or oversample <= 0:
        raise ValueError(f"oversample must be a positive integer, got {oversample}")
    dt = T / oversample
    t = np.arange(-span_T * T, span_T * T + dt, dt)
    return t, dt


def _normalize_energy_discrete(h: np.ndarray) -> np.ndarray:
    energy = np.sum(h**2)
    return h / np.sqrt(energy) if energy > 1e-12 else h


def _normalize_energy_continuous(h: np.ndarray, dt: float) -> np.ndarray:
    energy = np.trapz(h**2, dx=dt)
    return h / np.sqrt(energy) if energy > 1e-12 else h


def _normalize_amplitude(h: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(h))
    return h / max_val if max_val > 1e-12 else h


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



def compute_pulse(
    t: np.ndarray,
    name: str,
    alpha: float,
    T: float = 1.0,
    normalize: Optional[str] = None,
    dt: Optional[float] = None,
    return_energy: bool = False,
    **pulse_kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Generate a pulse shape by name, with optional normalization.

    Parameters
    ----------
    t : np.ndarray
        Time samples.
    name : str
        Name of the pulse registered in PULSE_FNS.
    alpha : float
        Roll-off factor.
    T : float, optional
        Symbol period (default: 1.0).
    normalize : {'discrete', 'continuous', 'amplitude'}, optional
        Normalization type. If None, no normalization.
    dt : float, optional
        Required if normalize is 'continuous'.
    return_energy : bool, optional
        If True, also return the energy of the unnormalized pulse.
    **pulse_kwargs : dict
        Additional pulse parameters (e.g., mu, epsilon).

    Returns
    -------
    h : np.ndarray
        Pulse samples.
    energy : float, optional
        Returned if return_energy=True.

    Raises
    ------
    ValueError
        If pulse name is unknown or normalization mode is invalid.
    """
    if name not in PULSE_FNS:
        raise ValueError(f"Unknown pulse '{name}'")

    h = PULSE_FNS[name](t, alpha=alpha, T=T, **pulse_kwargs)
    energy = None

    if normalize:
        if normalize == 'discrete':
            energy = np.sum(h**2)
            h = _normalize_energy_discrete(h)
        elif normalize == 'continuous':
            if dt is None:
                raise ValueError("'dt' required for continuous normalization; call t_axis first.")
            energy = np.trapz(h**2, dx=dt)
            h = _normalize_energy_continuous(h, dt)
        elif normalize == 'amplitude':
            energy = np.sum(h**2)
            h = _normalize_amplitude(h)
        else:
            raise ValueError(f"Unknown normalize mode '{normalize}'")

    if return_energy:
        if energy is None:
            energy = np.sum(h**2)
        return h, energy
    return h



def spectrum(h: np.ndarray, nfft: int = 2048, fs: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute the frequency-domain representation of a time-domain pulse.

    Parameters
    ----------
    h : np.ndarray
        Time-domain pulse samples.
    nfft : int, optional
        Number of FFT points (default: 2048).
    fs : float, optional
        Sampling frequency in Hz or normalized units (default: 1.0).

    Returns
    -------
    dict
        A dictionary containing:
        - 'fT' : Frequency axis normalized by the symbol rate T = 1/fs.
        - 'fB' : Frequency axis normalized by the two-sided bandwidth B = 1/(2T).
        - 'mag' : Magnitude spectrum |H(f)|.
        - 'mag_db' : Log-magnitude in dB scale (20·log₁₀|H(f)|).
        - 'H' : Complex FFT of the pulse.

    Notes
    -----
    The FFT result is normalized so that max(|H|) = 1.
    Both `fT` and `fB` are centered at zero using `fftshift`.
    """    
    
    H = np.fft.fft(h, nfft)
    H = H / np.max(np.abs(H))
    f = np.fft.fftfreq(nfft, d=1/fs)

    # Shift frequency and spectrum before magnitude calculations
    H_shifted = np.fft.fftshift(H)
    f_shifted = np.fft.fftshift(f)
    mag = np.abs(H_shifted)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))

    return {
        "fT": f_shifted,
        "fB": f_shifted * 2,
        "mag": mag,
        "mag_db": mag_db,
        "H": H_shifted,
    }


def get_pulse_info(
    pulse_name: str,
    alpha: float,
    span_T: float,
    *,
    T: float = 1.0,
    oversample: int = 200,
    normalize: str = "discrete",
    nfft: int = 2048,
    freq_axis: str = "fT",
    **pulse_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate the time- and frequency-domain information for a given pulse shape.

    Parameters
    ----------
    pulse_name : str
        Name of the pulse shape registered in PULSE_FNS (e.g., 'raised_cosine').
    alpha : float
        Roll-off or shaping parameter for the pulse.
    span_T : float
        Pulse half-span in symbol durations (total span = 2·span_T).
    T : float, optional
        Symbol duration in seconds (default: 1.0).
    oversample : int, optional
        Number of samples per symbol (default: 200).
    normalize : {'discrete', 'continuous', 'amplitude'}, optional
        Normalization method applied to the pulse (default: 'discrete').
    nfft : int, optional
        Number of FFT points for frequency-domain analysis (default: 2048).
    freq_axis : {'fT', 'fB'}, optional
        Choose between frequency axis normalized by symbol rate ('fT') or bandwidth ('fB').
    **pulse_kwargs : dict
        Additional parameters passed to the specific pulse generator (e.g., mu, gamma, epsilon).

    Returns
    -------
    dict
        {
            't'      : Time axis for impulse response.
            'h'      : Time-domain pulse samples.
            'f'      : Frequency axis (according to `freq_axis`).
            'mag'    : Magnitude spectrum |H(f)|.
            'mag_db' : Spectrum in dB (20·log₁₀|H(f)|).
        }

    Raises
    ------
    ValueError
        If `pulse_name` is not found or normalization type is invalid.
    """

    t, dt = t_axis(span_T, oversample, T)
    h = compute_pulse(
        t, pulse_name, alpha,
        T=T, dt=dt, normalize=normalize,
        **pulse_kwargs
    )
    spec = spectrum(h, nfft=nfft, fs=1/dt)
    return dict(t=t, h=h, f=spec[freq_axis], mag=spec["mag"], mag_db=spec["mag_db"])

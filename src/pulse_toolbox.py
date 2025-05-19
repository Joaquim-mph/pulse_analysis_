import numpy as np
from typing import Tuple, Optional, Dict, Union
from pulses import *

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


def _normalize_energy_continuous(h: np.ndarray, dt: float) -> np.ndarray:
    energy = np.trapz(h**2, dx=dt)
    return h / np.sqrt(energy) if energy > 1e-12 else h


def _normalize_amplitude(h: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(h))
    return h / max_val if max_val > 1e-12 else h



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
        if normalize == 'continuous':
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

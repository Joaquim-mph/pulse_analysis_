from typing import Literal, Union, Callable, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import upfirdn
import matplotlib.pyplot as plt
from pulse_toolbox import t_axis, compute_pulse, PULSE_FNS

def generate_qpsk_symbols(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    bits_i = rng.integers(0, 2, n, dtype=int)
    bits_q = rng.integers(0, 2, n, dtype=int)
    return (2 * bits_i - 1) + 1j * (2 * bits_q - 1)

def generate_bpsk_symbols(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    bits = rng.integers(0, 2, n)
    return 2 * bits - 1  # BPSK: ±1

def _resolve_pulse(pulse_ref: Union[str, Callable]) -> Callable:
    if callable(pulse_ref):
        return pulse_ref
    if pulse_ref in PULSE_FNS:
        return PULSE_FNS[pulse_ref]
    raise ValueError(f"Unknown pulse '{pulse_ref}'")

def eye_diagram(
    pulse: Union[str, Callable],
    *,
    alpha: float = 0.22,
    fs: int = 10,
    span_T: float = 10,
    n_symbols: int = 100_000,
    eye_T: float = 2.0,
    max_traces: int = 500,
    rng: Optional[np.random.Generator] = None,
    pulse_kwargs: Optional[dict] = None,
    parts: Sequence[str] = ("real",),
    show: bool = True,
    normalize: Literal["amplitude", "continuous", "discrete"] = "continuous"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build and (optionally) plot an eye diagram for any Nyquist‑I pulse.

    Returns
    -------
    eye : np.ndarray
        Array of shape (n_traces, eye_T*fs + 1) with complex eye samples.
    t_eye : np.ndarray
        Time axis (‑eye_T/2…+eye_T/2, including +eye_T/2).
    """
    g = _resolve_pulse(pulse)
    pulse_kwargs = pulse_kwargs or {}
    rng = rng or np.random.default_rng()
    a = generate_qpsk_symbols(n_symbols, rng=rng)

    t, dt = t_axis(span_T, fs, T=1.0)
    h = compute_pulse(
        t,
        name=pulse if isinstance(pulse, str) else pulse.__name__,
        alpha=alpha,
        T=1.0,
        normalize=normalize,
        dt=dt,
        **pulse_kwargs
    )

    s = upfirdn(h, a, fs)

    eye_span = int(eye_T * fs)
    n_tr     = min(max_traces, len(s) // eye_span)
    eye = s[:n_tr * eye_span].reshape(n_tr, eye_span)

    t_eye = np.linspace(-eye_T / 2, eye_T / 2, eye_span, endpoint=False)
    t_eye = np.append(t_eye, t_eye[-1] + 1/fs)

    last = eye[:, -1]
    second_last = eye[:, -2]
    extra = last + (last - second_last)
    eye = np.hstack([eye, extra[:, None]]) # Agrega un punto al final para asegurar t_eye tiene misma longitud que eye row

    if show:
        plt.figure(figsize=(8, 3 * len(parts)))
        for idx, part in enumerate(parts, 1):
            plt.subplot(len(parts), 1, idx)
            sig = eye.real if part == "real" else eye.imag
            plt.plot(t_eye, sig.T, color='k', lw=0.1)
            plt.title(f"Eye diagram ({part}) — {pulse if isinstance(pulse, str) else pulse.__name__}")
            plt.xlabel("t / T")
            plt.ylabel("Amplitude")
            plt.xlim(-eye_T / 2, eye_T / 2)
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    return eye, t_eye



def compute_max_distortion_analytical(
    pulse: Union[str, Callable],
    alpha: float,
    span_T: int = 10,
    T: float = 1.0,
    tau: float = 0.5,
    normalize: str = "continuous",
    **pulse_kwargs
) -> float:
    """
    Compute the maximum ISI distortion by analytically evaluating the pulse
    at interfering symbol positions offset by tau.

    Parameters
    ----------
    pulse : str or Callable
        Pulse shape (either a registered name or function).
    alpha : float
        Roll-off or shaping parameter.
    span_T : int
        Number of symbols to span (±span_T).
    T : float
        Symbol duration.
    tau : float
        Timing offset (as a multiple of T, default = 0.5).
    normalize : str
        Normalization method ('amplitude', 'discrete', 'continuous').
    **pulse_kwargs
        Additional arguments for the pulse function.

    Returns
    -------
    float
        Maximum ISI distortion: sum of absolute values of off-center samples.
    """
    pulse_fn = PULSE_FNS[pulse] if isinstance(pulse, str) else pulse

    # Time offsets (exclude k=0 to avoid main tap)
    k_vals = np.arange(-span_T, span_T + 1)
    k_vals = k_vals[k_vals != 0]
    t_isi = k_vals * T + tau * T

    # Evaluate pulse at ISI tap positions
    h_isi = pulse_fn(t_isi, alpha=alpha, T=T, **pulse_kwargs)

    # Normalize (optional, for fair comparison)
    if normalize == "amplitude":
        h_isi = h_isi / np.max(np.abs(pulse_fn(np.array([0.0]), alpha=alpha, T=T, **pulse_kwargs)))
    elif normalize == "discrete":
        t_grid = np.linspace(-span_T * T, span_T * T, 10000)
        h_grid = pulse_fn(t_grid, alpha=alpha, T=T, **pulse_kwargs)
        energy = np.sum(h_grid ** 2)
        h_isi = h_isi / np.sqrt(energy)
    elif normalize == "continuous":
        t_grid = np.linspace(-span_T * T, span_T * T, 10000)
        h_grid = pulse_fn(t_grid, alpha=alpha, T=T, **pulse_kwargs)
        dt = t_grid[1] - t_grid[0]
        energy = np.trapz(h_grid**2, dx=dt)
        h_isi = h_isi / np.sqrt(energy)

    # Compute max distortion (worst-case ISI)
    return np.sum(np.abs(h_isi))

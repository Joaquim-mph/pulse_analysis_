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


    eye_max_abs = np.max(np.abs(eye))    # magnitud completa
    # o, si sólo te interesa la parte real
    real_peak = np.max(np.abs(eye.real))

    return eye, t_eye, eye_max_abs, real_peak



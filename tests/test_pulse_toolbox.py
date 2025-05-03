# tests/test_pulse_toolbox.py
import sys, os
# Ensure src/ is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import numpy as np

from pulse_toolbox import (
    t_axis,
    compute_pulse,
    spectrum,
    PULSE_FNS,
    iplcp_freq
)

#------------------------------------------------------------------------------
# Fixtures and parametrization
#------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def params():
    """
    Shared parameters for tests.
    """
    return {
        'alpha': 0.22,
        'T': 1.0,
        'dt': 1.0/200,
        'sample_configs': [
            {'span_T': 5, 'oversample': 200},
            {'span_T': 0, 'oversample': 100},
        ]
    }

#------------------------------------------------------------------------------
# t_axis tests
#------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [{'span_T':5,'oversample':200}, {'span_T':2,'oversample':50}])
def test_t_axis_properties(cfg, params):
    span_T = cfg['span_T']
    oversample = cfg['oversample']
    T = params['T']
    t, dt = t_axis(span_T, oversample, T)
    # dt matches T/oversample
    assert dt == pytest.approx(T/oversample)
    # length and zero inclusion
    expected_len = int((2*span_T*T)/dt) + 1
    assert len(t) == expected_len
    assert np.isclose(t, 0.0).any()

def test_t_axis_invalid_oversample():
    with pytest.raises(ValueError):
        t_axis(span_T=1, oversample=0, T=1.0)

#------------------------------------------------------------------------------
# Pulse registry and DC gain
#------------------------------------------------------------------------------

def test_registered_pulses():
    required = {
        'raised_cosine', 'btrc', 'elp',
        'root_raised_cosine', 'iplcp_freq', 'iplcp_pulse'
    }
    missing = required - set(PULSE_FNS.keys())
    assert not missing, f"Missing pulses: {missing}"

@pytest.mark.parametrize("name,alpha,expected", [
    ('raised_cosine', 0.5, 1.0),
    ('btrc',          0.3, 1.0),
    ('elp',           0.1, 1.0)
])
def test_dc_gain_baseline(name, alpha, expected, params):
    # h(0) == expected (usually 1) for baseline pulses
    T = params['T']
    t, _ = t_axis(span_T=3, oversample=100, T=T)
    h = compute_pulse(t, name, alpha, normalize=None, T=T)
    idx0 = np.argmin(np.abs(t))
    assert h[idx0] == pytest.approx(expected, rel=1e-6)

@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_raised_cosine_edge_rolloff(alpha, params):
    # alpha=0 -> sinc; alpha=1 -> finite but wide
    T = params['T']
    t, _ = t_axis(3, 100, T)
    h = compute_pulse(t, 'raised_cosine', alpha, normalize=None, T=T)
    if alpha == 0.0:
        assert np.allclose(h, np.sinc(t/T), atol=1e-6)
    else:
        assert np.isfinite(h).all()

#------------------------------------------------------------------------------
# Normalization round-trip and error cases
#------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ['raised_cosine', 'btrc', 'elp', 'root_raised_cosine', 'iplcp_pulse'])
def test_roundtrip_amplitude_normalization(name, params):
    T, alpha, dt = params['T'], params['alpha'], params['dt']
    t, _ = t_axis(3, 200, T)
    # baseline
    h0 = compute_pulse(
        t, name, alpha,
        normalize=None, T=T, dt=dt,
        **({'mu':1.6,'gamma':1,'epsilon':0.1} if 'iplcp' in name else {})
    )
    # amplitude normalize and reverse
    h_amp = compute_pulse(
        t, name, alpha,
        normalize='amplitude', T=T, dt=dt,
        **({'mu':1.6,'gamma':1,'epsilon':0.1} if 'iplcp' in name else {})
    )
    peak = np.max(np.abs(h0))
    assert np.allclose(h_amp * peak, h0, atol=1e-6)

def test_continuous_without_dt_raises(params):
    T, alpha = params['T'], params['alpha']
    t, _ = t_axis(3, 100, T)
    with pytest.raises(ValueError):
        compute_pulse(t, 'raised_cosine', alpha, normalize='continuous', T=T)

#------------------------------------------------------------------------------
# Spectrum tests
#------------------------------------------------------------------------------

@pytest.mark.parametrize("pulse,nulls", [
    ('raised_cosine', [(1+0.22)/2, -(1+0.22)/2])
])
def test_nulls_in_spectrum(pulse, nulls, params):
    T, alpha, dt = params['T'], params['alpha'], params['dt']
    t, _ = t_axis(5, 200, T)
    h = compute_pulse(
        t, pulse, alpha,
        normalize='discrete', T=T, dt=dt,
        **({'mu':1.6,'gamma':1,'epsilon':0.1} if 'iplcp' in pulse else {})
    )
    f, mag, _, _ = spectrum(h, nfft=8192, normalise=True, fs=1/dt)
    for nf in nulls:
        idx = np.argmin(np.abs(f - nf))
        assert mag[idx] < 5e-3, f"{pulse} null at {nf} failed: {mag[idx]}"

# Coverage for iplcp_freq at non-zero frequency

def test_iplcp_freq_first_sidelobe(params):
    T, alpha = params['T'], params['alpha']
    # At first sidelobe (approx) the magnitude should be strictly less than 1
    f_test = 1/alpha
    H_val = iplcp_freq(
        np.array([f_test]), alpha=alpha,
        mu=1.6, gamma=1, epsilon=0.1, T=T
    )[0]
    assert H_val < 1.0, f"Expected |H| < 1 at f={f_test}, got {H_val}"

#------------------------------------------------------------------------------
# Invalid pulse name
#------------------------------------------------------------------------------

def test_compute_pulse_invalid(params):
    t, _ = t_axis(1, 10, params['T'])
    with pytest.raises(ValueError):
        compute_pulse(t, 'INVALID', params['alpha'], T=params['T'])

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))

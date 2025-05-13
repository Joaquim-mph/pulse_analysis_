# Pulse Shape Analysis for Digital Communication Systems

This project provides a comprehensive simulation framework for analyzing pulse shaping in digital communication systems. It includes:

- Time-domain and frequency-domain characterization of pulse shapes
- Eye diagram visualization for various modulation formats
- Closed-form Bit Error Rate (BER) analysis under Inter-Symbol Interference (ISI) and Co-Channel Interference (CCI)
- Support for classical and novel pulse shapes (Raised Cosine, BTRC, ELP, IPLCP)

## Features

### Pulse Shapes Implemented
- **Raised Cosine (RC)**
- **Better-than-Raised-Cosine (BTRC)**
- **Exponential-Linear Pulse (ELP)**
- **Improved Parametric Linear-Combination Pulse (IPLCP)**

Each pulse is implemented with proper parameterization (e.g., α, μ, β, ε), with time-domain truncation and energy normalization.

### Visualizations
- Impulse response and frequency response with parameter sweep
- Eye diagrams with QPSK/BPSK input
- BER vs. timing offset and SNR/SIR with closed-form expressions from Beaulieu’s method

### BER Analysis
Includes ISI-only, CCI-only, and joint ISI+CCI models with tunable:
- Symbol timing offsets (τ/T)
- Roll-off factor α
- SNR and SIR levels
- Number of interference sources (L)

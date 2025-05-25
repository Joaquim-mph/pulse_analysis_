import numpy as np
import matplotlib.pyplot as plt
from styles import set_plot_style
import scienceplots

set_plot_style("ink_sketch")

def plot_psk_constellation(M, title, ax, radius=1.0, rotation=0.0):
    """Plot an M-PSK constellation diagram with optional rotation (in radians)."""
    angles = 2 * np.pi * np.arange(M) / M + rotation
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    labels = [format(i, f'0{int(np.log2(M))}b') for i in range(M)]

    # Constellation points
    ax.scatter(x, y, c='black', zorder=3)
    for i in range(M):
        ax.text(x[i] * 1.2, y[i] * 1.3, f'"{labels[i]}"', color='blue',
                fontsize=10, ha='center', va='center')
        ax.text(x[i] * 0.85, y[i] * 0.85, f'$S_{{{i+1}}}$', fontsize=12,
                ha='center', va='center')

    # Dashed circle
    circle = plt.Circle((0, 0), radius, color='gray', linestyle='dotted',
                        fill=False, linewidth=1, zorder=1)
    ax.add_artist(circle)

    # Axes and formatting
    ax.axhline(0, color='gray', lw=1, alpha=0.1)
    ax.axvline(0, color='gray', lw=1, alpha=0.1)
    ax.set_title(f'{title} (M={M})', color='red', fontsize=12)
    ax.set_xlabel(r'$\psi_1(t)$')
    ax.set_ylabel(r'$\psi_2(t)$')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.5*radius, 1.5*radius)
    ax.set_ylim(-1.5*radius, 1.5*radius)



fig, axs = plt.subplots(1, 3, figsize=(14, 4))

# No rotation
plot_psk_constellation(2, 'BPSK', axs[0], radius=1.0)
# QPSK rotated 45 degrees (π/4)
plot_psk_constellation(4, 'QPSK', axs[1], radius=1.0, rotation=np.pi/4)
# 8PSK default
plot_psk_constellation(8, '8PSK', axs[2], radius=1.0)


plt.tight_layout()
plt.savefig("figures/m_psk.png", dpi = 300)
#plt.show()



# Data
freq = [230, 245, 260, 275, 290, 305, 320, 335, 350, 365, 380, 395, 410, 425, 440, 455, 470, 485, 500, 515, 530]
ber  = [4.823e-3, 6.2525e-6, 4.6417e-1, 5.0345e-1, 3.4222e-1, 5.0170e-1,
        2.6596e-3, 3.6168e-3, 1.8685e-1, 6.8454e-3, 4.2883e-3, 7.5347e-5,
        2.6213e-5, 3.5779e-5, 1.0752e-4, 1.2298e-4, 9.5208e-6, 1.7036e-4,
        1.7193e-4, 2.0001e-6, 4.7379e-4]

# Plot
plt.figure(figsize=(9, 3))
plt.semilogy(freq, ber, marker='D', linestyle='--')
#plt.title('BER vs Frequency')
plt.xlabel('Frequency (MHz)')
plt.ylabel('BER (log scale)')
#plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("figures/ber_frec_experimental.png", dpi = 300)

import matplotlib.pyplot as plt

# Data
cci_db = [40, 45, 50, 55, 60, 65, 70]
ber =    [5.16798e-6, 4.1187e-5, 5.1755e-2, 
          8.2407e-2, 1.2427e-1, 2.2221e-1, 4.9767e-1]

# Plot
plt.figure(figsize=(8, 4))
plt.semilogy(cci_db, ber, marker='D', linestyle='-', linewidth=1.5)
plt.xlabel('Ganancia interferidor (dB)')
plt.ylabel('BER (log scale)')
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("figures/interference_experimental.png", dpi = 300)
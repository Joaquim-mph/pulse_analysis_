import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
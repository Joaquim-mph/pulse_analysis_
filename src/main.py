# main.py
"""
Genera comparaciones de pulsos Nyquist y diagramas de ojo.
"""
import os
import argparse
import logging
from pulse_toolbox import get_pulse_info
from eye_utils import eye_diagram
from plot_utils import plot_pulse_markers, plot_eye_traces
from styles import set_plot_style


# logging
default_level = logging.INFO
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=default_level
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparativa de pulsos Nyquist y análisis de diagramas de ojo"
    )
    parser.add_argument(
        "--alphas", nargs='+', type=float,
        default=[0.22, 0.50],
        help="Valores de roll-off alpha a comparar"
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures",
        help="Directorio base para guardar las figuras"
    )
    return parser.parse_args()


def generate_pulse_data(alpha, span_T, T, ovs, nfft, normalize, freq_axis):
    logger.debug("Generando datos de pulsos para α=%.2f", alpha)
    pulse_specs = [
        ("Raised Cosine", "raised_cosine", {}),
        ("BTRC", "btrc", {}),
        ("ELP β=0.1", "elp", dict(beta=0.1)),
        ("IPLCP μ=1.6 ε=0.1", "iplcp", dict(mu=1.6, gamma=1, epsilon=0.1)),
    ]
    data = []
    for label, key, extra in pulse_specs:
        logger.debug("  - Pulsado: %s", label)
        info = get_pulse_info(
            key, alpha, span_T,
            T=T, oversample=ovs, nfft=nfft,
            normalize=normalize, freq_axis=freq_axis,
            **extra
        )
        data.append((label, info['t'], info['h'], info['f'], info['mag'], info['mag_db']))
    return data


def main():
    args = parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    

    # Parámetros globales
    span_T = 10
    ovs = 20
    nfft = 2048
    T = 1.0
    normalize = "amplitude"
    freq_axis = "fB"

    # Comparativa de pulsos en tiempo/frecuencia para cada alpha
    pulsos_dir = os.path.join(out_dir, "pulses_comparation")
    os.makedirs(pulsos_dir, exist_ok=True)
    for alpha in args.alphas:
        set_plot_style("prism_rain")
        logger.info("Procesando pulsos para α=%.2f", alpha)
        pulse_data = generate_pulse_data(alpha, span_T, T, ovs, nfft, normalize, freq_axis)
        prefix = os.path.join(pulsos_dir, f"pulse_compare_{int(alpha*100):03d}")
        plot_pulse_markers(
            pulse_data,
            t_xlim=(0, 4), f_xlim=(0, 5), f_mag_xlim=(0, 3),
            prefix=prefix,
            show=False,
            figsize=(7, 7), markersize=3, linewidth=0.7,
            db_ylim=(-200, 5), freq_axis_label="f/B",
            f_db_xlim=(-10, 10)
        )
        logger.info("Guardado comparativa de pulsos: %s*", prefix)

        # ================================================
        # Graficar cada pulso individualmente con t_lim = ±5T
        indiv_dir = os.path.join(pulsos_dir, "individual")
        os.makedirs(indiv_dir, exist_ok=True)
        for label, t, h, f, mag, mag_db in pulse_data:
            set_plot_style("matlab")
            logger.info("Graficando pulso individual: %s, α=%.2f", label, alpha)
            prefix_ind = os.path.join(
                indiv_dir,
                f"{label.replace(' ', '_').lower()}_{int(alpha*100):03d}"
            )
            plot_pulse_markers(
                [(label, t, h, f, mag, mag_db)],
                prefix=prefix_ind,
                show=False,
                savefig=True,
                which="impulse",
                t_xlim=(-4, 4)
            )
            logger.info("Guardado pulso individual: %s", prefix_ind)
        # ================================================



    # Estilo para diagramas de ojo
    logger.info("Aplicando estilo 'prism_rain'")
    set_plot_style("prism_rain")

    # Generar y guardar diagramas de ojo y métricas
    eyes_dir = os.path.join(out_dir, "eyes_diagrams")
    os.makedirs(eyes_dir, exist_ok=True)
    for alpha in args.alphas:
        logger.info("Generando diagramas de ojo para α=%.2f", alpha)
        suffix = f"alpha{int(alpha*100):03d}"
        # Raised Cosine
        rc_eye, rc_t, rc_max, rc_open = eye_diagram(
            "raised_cosine", alpha=alpha,
            normalize="continuous", fs=10,
            span_T=6, eye_T=2.0,
            n_symbols=100_000, max_traces=500
        )
        logger.debug("RC ISI_max=%.4f, Eye_open=%.4f", rc_max, rc_open)

        # Pulsos adicionales
        pulses = [
            ("btrc", {}),
            ("elp", dict(beta=0.1)),
            ("iplcp", dict(mu=1.6, gamma=1, epsilon=0.1)),
        ]
        # Graficar todos
        logger.info("Procesando pulso rc")
        plot_eye_traces(
            eye_data=rc_eye, t_eye=rc_t,
            pulse="raised_cosine", alpha=alpha,
            parts=("real",), prefix=os.path.join(eyes_dir, f"rc_{suffix}"),
            show=False
        )
        for name, kwargs in pulses:
            logger.info("Procesando pulso %s", name)
            eye_data, t_eye, max_val, open_val = eye_diagram(
                name, alpha=alpha, pulse_kwargs=kwargs,
                normalize="continuous", fs=10,
                span_T=6, eye_T=2.0,
                n_symbols=100_000, max_traces=500
            )
            logger.debug("%s ISI_max=%.4f, Eye_open=%.4f", name.upper(), max_val, open_val)
            plot_eye_traces(
                eye_data=eye_data, t_eye=t_eye, pulse=name,
                alpha=alpha, pulse_kwargs=kwargs,
                parts=("real",), prefix=os.path.join(eyes_dir, f"{name}_{suffix}"),
                show=False
            )

        logger.info("Maximum Amplitude Values from Eye Diagrams (α = %.2f):", alpha)
        logger.info("%10s | %18s | %15s", 'Pulse', 'Max complex Amp', 'Max real Amp')
        logger.info('%s', '-' * 56)
        logger.info("%10s | %18.4f | %15.4f", 'RC', rc_max, rc_open)
        for name, kwargs in pulses:
            _, _, max_val, open_val = eye_diagram(
                name, alpha=alpha, pulse_kwargs=kwargs,
                normalize="continuous", fs=10,
                span_T=6, eye_T=2.0,
                n_symbols=100_000, max_traces=500
            )
            logger.info("%10s | %18.4f | %15.4f", name.upper(), max_val, open_val)


if __name__ == "__main__":
    main()


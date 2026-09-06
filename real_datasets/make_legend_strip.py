"""Standalone legend strip for the main-text and SI real-dataset composites.

The composites (numerics.pdf, more-numerics.pdf) are assembled outside this
repository from the per-dataset two-panel figures; this script renders the
single shared legend as its own PDF, from the same style dictionaries as the
panels (sweep_utils), so the two cannot drift apart. The strip is sized to
span two side-by-side panel figures (each 6 in wide) plus their gap, so it
can be placed under the plot rows at the same scale as the panels.

    python make_legend_strip.py --classical-alpha 0.6 --out numerics_legend.pdf
"""

import argparse

import matplotlib.pyplot as plt
import sweep_utils

sweep_utils.apply_plot_style()


def make_legend_strip(keys, classical_alpha, out, width_in=12.5, fontsize=12):
    fig = plt.figure(figsize=(width_in, 0.45))
    handles = [sweep_utils.legend_handle(k, alpha=classical_alpha) for k in keys]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.4,
        columnspacing=2.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Render the shared legend strip.")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(sweep_utils.MAIN_LEGEND_ORDER),
        help="legend entries in order (method keys of sweep_utils)",
    )
    parser.add_argument(
        "--classical-alpha",
        type=float,
        default=1.0,
        help="opacity of the classical entries (match the panels' --classical-alpha)",
    )
    parser.add_argument("--fontsize", type=float, default=12)
    parser.add_argument("--width", type=float, default=12.5, help="strip width in inches")
    parser.add_argument("--out", type=str, default="numerics_legend.pdf")
    args = parser.parse_args()
    make_legend_strip(args.keys, args.classical_alpha, args.out, args.width, args.fontsize)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    parser = argparse.ArgumentParser(
        description="Plot mean fitness vs. agent count from a CSV file"
    )
    parser.add_argument(
        "csvfile",
        help="Path to CSV file with two columns: N (agent count), fitness"
    )
    parser.add_argument(
        "-o", "--out",
        help="If given, save plot to this file (PDF/SVG/PNG) instead of showing it"
    )
    args = parser.parse_args()

    # ——— Load data ———
    # Assumes header row, so skiprows=1
    data = np.loadtxt(args.csvfile, delimiter=",", skiprows=1)
    Ns = data[:, 0].astype(int)
    fitnesses = data[:, 1]

    # ——— Matplotlib configuration ———
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Palatino", "serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with circular markers, about 20 evenly spaced
    ax.plot(Ns, fitnesses, marker='o', markevery=max(len(Ns)//20, 1))
    ax.set_title("Fitness vs. Number of Agents (N)")
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Average Fitness Score")
    ax.grid(True, which='both')
    # ax.minorticks_on()
    ax.set_xticks(range(3, 16))

    plt.tight_layout()

    if args.out:
        fmt = args.out.rsplit('.', 1)[-1].lower()
        if fmt in ['pdf', 'svg']:
            plt.savefig(args.out, bbox_inches='tight')
        else:
            plt.savefig(args.out, bbox_inches='tight', dpi=300)
        print(f"Saved high-quality plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

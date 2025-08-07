#!/usr/bin/env python3
import re
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

def parse_fitness(logfile_path):
    """
    Parses lines like:
      Epoch   1:      0.0599:     0.0599 | Neurons:  16 Synapses:  19 | …
    and returns two lists: epochs, fitnesses.
    """
    epochs, fitnesses = [], []
    pattern = re.compile(r"Epoch\s+(\d+):\s+([0-9.]+)")
    with open(logfile_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                fitnesses.append(float(m.group(2)))
    return epochs, fitnesses

def main():
    parser = argparse.ArgumentParser(
        description="Plot fitness over epochs from a training.log file"
    )
    parser.add_argument("logfile", help="Path to training.log")
    parser.add_argument(
        "-o", "--out", help="If given, save plot to this file (PDF/SVG/PNG) instead of showing it"
    )
    args = parser.parse_args()

    epochs, fitnesses = parse_fitness(args.logfile)
    if not epochs:
        print(f"No epochs found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

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

    # Plot with a solid line and circular markers every N points
    ax.plot(epochs, fitnesses, marker='o', markevery=max(len(epochs)//20,1))
    ax.set_title("Fitness over Training Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fitness")
    ax.grid(True, which='both')
    ax.minorticks_on()

    plt.tight_layout()

    if args.out:
        # Save vector format if possible
        fmt = args.out.split('.')[-1].lower()
        if fmt in ['pdf', 'svg']:
            plt.savefig(args.out, bbox_inches='tight')
        else:
            plt.savefig(args.out, bbox_inches='tight', dpi=300)
        print(f"Saved high‑quality plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()


# cd ~/neuromorphic/turtwig/results_sim/hopper/250423/tenn2/n-17/250423-031032-connorsim_snn_eons-v01
# python ../plot_fit.py training.log  

# cd 250729-085053-connorsim_snn_eons-v01
# cd 250729-085104-connorsim_snn_eons-v01
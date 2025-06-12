#!/usr/bin/env python3
import re
import sys
import argparse
import matplotlib.pyplot as plt

def parse_fitness(logfile_path):
    """
    Parses lines like:
      Epoch   1:      0.0599:     0.0599 | Neurons:  16 Synapses:  19 | …
    and returns two lists: epochs, fitnesses.
    """
    epochs = []
    fitnesses = []
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
        "-o", "--out", help="If given, save plot to this file instead of showing it"
    )
    args = parser.parse_args()

    epochs, fitnesses = parse_fitness(args.logfile)
    if not epochs:
        print(f"No epochs found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, fitnesses, marker='o', linestyle='--')
    plt.title("Fitness over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

# cd ~/neuromorphic/turtwig/results_sim/hopper/250423/tenn2/n-17/250423-031032-connorsim_snn_eons-v01
# python ~/neuromorphic/turtwig/results_sim/plot_fit.py training.log
#!/usr/bin/env python3
import re, sys, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

def parse_training_log(p):
    epochs, fitnesses, neurons, synapses = [], [], [], []
    pat = re.compile(r"Epoch\s+(\d+):\s+([0-9.]+).*?\|\s*Neurons:\s*(\d+)\s*Synapses:\s*(\d+)\s*\|")
    pat_fallback = re.compile(r"Epoch\s+(\d+):\s+([0-9.]+)")
    with open(p, "r") as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                fitnesses.append(float(m.group(2)))
                neurons.append(int(m.group(3)))
                synapses.append(int(m.group(4)))
            else:
                mf = pat_fallback.search(line)
                if mf:
                    epochs.append(int(mf.group(1)))
                    fitnesses.append(float(mf.group(2)))
                    neurons.append(None)
                    synapses.append(None)
    have_struct = [i for i,(n,s) in enumerate(zip(neurons, synapses)) if n is not None and s is not None]
    return {
        "epochs": epochs,
        "fitnesses": fitnesses,
        "epochs_ns": [epochs[i] for i in have_struct],
        "neurons": [neurons[i] for i in have_struct],
        "synapses": [synapses[i] for i in have_struct],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--out-prefix", help="Save plots as <prefix>_fitness.png/_neurons.png/_synapses.png")
    args = ap.parse_args()

    d = parse_training_log(args.logfile)
    if not d["epochs"]:
        print("No epochs parsed."); sys.exit(1)

    mpl.rcParams.update({"figure.dpi":300, "savefig.dpi":300, "font.size":14, "lines.linewidth":2.0})

    figs = []

    # 1) Fitness
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(d["epochs"], d["fitnesses"])
    ax1.set_title("Fitness over Training Epochs"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Fitness")
    ax1.grid(True); ax1.set_ylim(0,1); fig1.tight_layout(); figs.append(("fitness", fig1))

    # 2) Neurons
    if d["epochs_ns"]:
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(d["epochs_ns"], d["neurons"], marker="o")
        ax2.set_title("Number of Neurons over Epochs"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Neurons")
        ax2.grid(True); fig2.tight_layout(); figs.append(("neurons", fig2))
    else:
        print("No neuron counts found; skipping neurons plot.", file=sys.stderr)

    # 3) Synapses
    if d["epochs_ns"]:
        fig3, ax3 = plt.subplots(figsize=(10,6))
        ax3.plot(d["epochs_ns"], d["synapses"], marker="o")
        ax3.set_title("Number of Synapses over Epochs"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Synapses")
        ax3.grid(True); fig3.tight_layout(); figs.append(("synapses", fig3))
    else:
        print("No synapse counts found; skipping synapses plot.", file=sys.stderr)

    if args.out_prefix:
        for name, fig in figs:
            fig.savefig(f"{args.out_prefix}_{name}.png", bbox_inches="tight")
            print(f"saved: {args.out_prefix}_{name}.png")
        # If saving, I can still show interactively if I want:
        # plt.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()






# cd ~/neuromorphic/turtwig/results_sim/hopper/250423/tenn2/n-17/250423-031032-connorsim_snn_eons-v01
# python ../plot_fit.py training.log  

# cd 250729-085053-connorsim_snn_eons-v01
# cd 250729-085104-connorsim_snn_eons-v01
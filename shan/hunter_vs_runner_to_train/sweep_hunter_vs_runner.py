#!/usr/bin/env python3
"""
sweep_hunter_vs_runner.py

Run hunter_vs_runner.py over a range of agent counts (N), let it save each per‐N fitness plot
(via --save_plot), bypass the “select a project” menu by explicitly supplying your project
folder (or network.json), and then produce a summary fitness‐vs‐N plot.

Usage:
    python sweep_hunter_vs_runner.py \
      --N_low 3 --N_high 14 \
      --root /path/to/results \
      --project 250525-052430-connorsim_snn_eons-v01 \
      --world_yaml ./world_bin.yaml \
      --cy 2000 \
      --trial_seed 203 \
      --trials 100
"""
import os
import re
import sys
import pty
import argparse
import csv

# headless backend for the final summary plot
import matplotlib
# matplotlib.use("Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def run_for_N(N, script_path, root, project, world_yaml, cy, trial_seed, trials):
    cmd = [
        sys.executable,
        script_path,
        "run",
    ]

    # ** either** supply the project folder name (under --root)
    if project:
        cmd.append(project)

    # now the usual flags
    cmd += [
        "--root",       root,
        "--world_yaml", world_yaml,
        "--cy",         str(cy),
        "--save_plot",
        "--trial_seed", str(trial_seed),
        "-T",           str(trials),
        "-N",           str(N),
    ]

    print(f"\n>> [{N}] Spawning: {' '.join(cmd)}")

    buf = []
    def _master_read(fd):
        data = os.read(fd, 1024)
        if not data:
            return data
        buf.append(data)
        # os.write(sys.stdout.fileno(), data)
        return data

    exit_status = pty.spawn(cmd, _master_read)
    out = b"".join(buf).decode("utf-8", errors="ignore")

    # parse final average fitness
    m = re.search(r"Fitness after\s*\d+\s*trials:\s*([0-9]+\.[0-9]+)", out)
    if m:
        return float(m.group(1))
    all_fits = re.findall(r"Overall Fitness:\s*([0-9]+\.[0-9]+)", out)
    return float(all_fits[-1]) if all_fits else 0.0


def main():
    parser = argparse.ArgumentParser(description="Sweep hunter_vs_runner.py over N")
    parser.add_argument("--N_low",      type=int, required=True, help="minimum N")
    parser.add_argument("--N_high",     type=int, required=True, help="maximum N")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--project",
                     help="Name of the project folder under --root (bypasses menu)")

    parser.add_argument("--root",       type=str, default="out",
                        help="the same --root you’d pass to hunter_vs_runner.py")
    parser.add_argument("--world_yaml", type=str, default="./world.yaml", help="--world_yaml")
    parser.add_argument("--cy",         type=int, default=2000,          help="--cy")
    parser.add_argument("--trial_seed", type=int, default=12345,         help="--trial_seed")
    parser.add_argument("--trials", "-T", type=int, default=10,          help="-T trials")

    args = parser.parse_args()

    # find hunter_vs_runner.py next to this script
    script_path = os.path.join(os.path.dirname(__file__), "hunter_vs_runner.py")
    if not os.path.isfile(script_path):
        print(f"Error: cannot find {script_path}", file=sys.stderr)
        sys.exit(1)

    Ns = list(range(args.N_low, args.N_high + 1))
    final_scores = []
    total = len(Ns)

    print(f"Sweeping N from {args.N_low} to {args.N_high} ({total} runs)…")
    for idx, N in enumerate(Ns, start=1):
        final = run_for_N(
            N,
            script_path,
            args.root,
            args.project,
            args.world_yaml,
            args.cy,
            args.trial_seed,
            args.trials
        )
        final_scores.append(final)
        print(f"[{idx}/{total}] → N={N}: final avg fitness = {final:.4f}")

    # make summary plot
    out_dir = os.path.join("./plots", args.project)
    os.makedirs(out_dir, exist_ok=True)
    summary_plot = os.path.join(
        out_dir,
        f"fitness_vs_N_{args.N_low}_{args.N_high}.png"
    )

    plt.figure()
    plt.plot(Ns, final_scores, marker='o')
    plt.xlabel("Number of agents (N)")
    plt.ylabel("Average Fitness")
    plt.title(f"Fitness vs N")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_plot)
    print(f"\n✓ Sweep complete. Summary plot → {summary_plot}")
    data_file = os.path.join(out_dir, "fitness_vs_N_data.csv")
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "fitness"])
        writer.writerows(zip(Ns, final_scores))

    plt.show()


if __name__ == "__main__":
    main()






# python sweep_hunter_vs_runner.py --N_low 6 --N_high 8 --root ~/neuromorphic/turtwig/results_sim/hopper/250525/farp/6-wtp/ --project 250525-224042-connorsim_snn_eons-v01 --world_yaml ./world_bin.yaml --cy 2000 --trial_seed 203 --trials 2
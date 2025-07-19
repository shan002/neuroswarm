#!/usr/bin/env python3
"""
sweep_phase_diagram_range.py

Sweep a single CLI flag (e.g. -N) from MIN to MAX (inclusive), with STEP,
running TRIALS independent runs at each point, showing live convergence
and then a final phase-diagram of fitness vs flag. Also passes a trained
network file via --root to the target script.
"""
import argparse
import subprocess
import shlex
import re
import sys
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep a CLI flag over a numeric range and plot fitness."
    )
    p.add_argument('--param', '-p', type=str, default='-N',
                   help="Flag to sweep (e.g. -N or --agents)")
    p.add_argument('--min', type=int, required=True,
                   help="Minimum value of the flag")
    p.add_argument('--max', type=int, required=True,
                   help="Maximum value of the flag")
    p.add_argument('--step', type=int, default=1,
                   help="Step size (default: 1)")
    p.add_argument('--trials', '-T', type=int, default=10,
                   help="Number of independent runs per value")
    p.add_argument('--root', type=str, required=True,
                   help="Path to trained network file (passed as --root)")
    p.add_argument('--script', '-s', type=str, default='hunter_vs_runner_wtp.py',
                   help="Path to the target script")
    p.add_argument('extra_args', nargs=argparse.REMAINDER,
                   help="Additional flags to pass to the target script (after a literal --)")
    return p.parse_args()

def run_single_trial(script, flag, value, root, extra):
    cmd = [sys.executable, script, 'run',
           flag, str(value),
           '-T', '1',
           '--root', root]
    # strip leading '--' if present
    if extra and extra[0] == '--':
        extra = extra[1:]
    cmd += extra

    print(f"> {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Script failed (exit {proc.returncode})")

    m = re.search(r"Fitness after\s*\d+\s*trials:\s*([0-9.]+)", proc.stdout)
    if not m:
        print(proc.stdout, file=sys.stderr)
        raise ValueError("Couldn't parse fitness from output")
    return float(m.group(1))

def main():
    args = parse_args()

    # build the list of swept values
    values = list(range(args.min, args.max + 1, args.step))
    phase_results = {}

    for v in values:
        fitnesses = []

        # set up live convergence plot for this v
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel("Trial")
        ax.set_ylabel("Fitness")
        ax.set_title(f"Convergence for {args.param} = {v}")
        line, = ax.plot([], [], marker='o')
        plt.show()

        # run each trial one by one
        for t in range(1, args.trials + 1):
            fit = run_single_trial(
                args.script,
                args.param,
                v,
                args.root,
                args.extra_args
            )
            fitnesses.append(fit)

            # update plot
            line.set_data(list(range(1, t + 1)), fitnesses)
            ax.relim(); ax.autoscale()
            plt.draw(); plt.pause(0.1)

        plt.ioff()
        plt.close(fig)

        phase_results[v] = fitnesses[-1]  # record final fitness

    # final phase-diagram
    xs = sorted(phase_results)
    ys = [phase_results[x] for x in xs]
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel(args.param)
    plt.ylabel("Final Fitness")
    plt.title("Phase Diagram")
    plt.show()

    # print summary
    print("\nSweep complete. Final fitness values:")
    for x in xs:
        print(f"  {args.param} = {x} → {phase_results[x]:.4f}")

if __name__ == "__main__":
    main()

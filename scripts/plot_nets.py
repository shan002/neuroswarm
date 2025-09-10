#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import signal

def main():
    # Expand ~
    VIZ_DIR     = os.path.expanduser('~/neuromorphic/framework/viz')
    CONFIG_FILE = os.path.expanduser('~/neuromorphic/framework/viz/config/config.json')

    if len(sys.argv) < 2:
        print("Usage: python plot_nets.py <network.json>")
        sys.exit(1)
    target_dir = os.path.expanduser(sys.argv[1])
    target_network = os.path.join(target_dir, 'best.json')

    # Start simulation process
    os.chdir(os.path.expanduser('~/neuromorphic/turtwig'))
    sim_cmd = [
        sys.executable, 'experiment_tenn2.py',
        'run', '--cy', '-1', '-N', '6',
        '--network', target_network,
        '--all_counts_stream', '{"source":"serve","port":8100}', '--sta', '--caspian'
    ]
    print("Starting simulation:", " ".join(sim_cmd))
    sim_proc = subprocess.Popen(sim_cmd)

    # Give the sim a moment to bind port 8100
    time.sleep(1)

    # Start viz process
    viz_cmd = [
        'love', VIZ_DIR,
        '--config', CONFIG_FILE,
        '-n', target_network,
        '--show_input_id',
        '--show_output_id',
        '-i', '{"source":"request","port":"8100","host":"localhost"}',
    ]

    # '--remove_unnecessary_neuron'
    print("Starting viz:", " ".join(viz_cmd))
    viz_proc = subprocess.Popen(viz_cmd)

    # Wait for viz to finish, then tear down sim
    try:
        retcode = viz_proc.wait()
    except KeyboardInterrupt:
        print("Interrupted! Stopping both processes…")
        viz_proc.send_signal(signal.SIGINT)
        retcode = 1

    # Make sure we clean up the sim process
    if sim_proc.poll() is None:  # still running?
        print("Stopping simulation process")
        sim_proc.terminate()
        sim_proc.wait()

    print(target_network.replace('best.json', 'training.log'))

    fit_cmd = [
        sys.executable, 'scripts/plot_fit.py',
        os.path.join(target_dir, 'training.log'),
    ]
    print("Starting fitness plotting:", " ".join(fit_cmd))
    fit_proc = subprocess.Popen(fit_cmd)
    fit_proc.wait()
    print("Plotting fitness done")

    sys.exit(retcode)

if __name__ == "__main__":
    main()


## To simulate and visuzalize I need to run the following commands
# cd ~/neuromorphic/turtwig
# python experiment_tenn2.py run --cy 1000 -N 6 --network ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-132733-connorsim_snn_eons-v01/best.json --all_counts_stream '{"source":"serve","port":8100}'
# love ~/neuromorphic/framework/viz --config ~/neuromorphic/framework/viz/config/config.json -n ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-132733-connorsim_snn_eons-v01/best.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}' --remove_unnecessary_neuron
# python ~/neuromorphic/turtwig/scripts/plot_fit.py ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-132733-connorsim_snn_eons-v01/training.log

## With this script I can do the same with the following single command
# python ~/neuromorphic/turtwig/scripts/plot_nets.py ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-132733-connorsim_snn_eons-v01
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
    target_network = os.path.expanduser(sys.argv[1])

    # Start simulation process
    os.chdir(os.path.expanduser('~/neuromorphic/turtwig'))
    sim_cmd = [
        sys.executable, 'experiment_tenn2.py',
        'run', '--cy', '1000', '-N', '6',
        '--network', target_network,
        '--all_counts_stream', '{"source":"serve","port":8100}', '--sta'
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
        '--remove_unnecessary_neuron'
    ]
    print("Starting viz:", " ".join(viz_cmd))
    viz_proc = subprocess.Popen(viz_cmd)

    # 4) Wait for viz to finish, then tear down sim
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

    sys.exit(retcode)

if __name__ == "__main__":
    main()



# cd ~/neuromorphic/turtwig

# python experiment_tenn2.py run --cy 1000 -N 6 --network ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-085053-connorsim_snn_eons-v01/best.json --all_counts_stream '{"source":"serve","port":8100}'

# love ~/neuromorphic/framework/viz --config ~/neuromorphic/framework/viz/config/config.json -n ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-085104-connorsim_snn_eons-v01/best.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}' --remove_unnecessary_neuron

# python plot_nets.py ~/neuromorphic/turtwig/results_sim/hopper/250804/tenn2/n-6/250729-085104-connorsim_snn_eons-v01/best.json
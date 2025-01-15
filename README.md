# Kevin's Neuromorphic Stuff

To install, you'll need to have a python virtual environment setup first.

Experiments 1-5 require Tennlab's framework for now, so install that first and use its default environment (`source framework/pyframework/bin/activate`)

`experiment_tenn2.py` requires [my fork](https://github.com/kenblu24/RobotSwarmSimulator/tree/main) of [Connor Mattson's RobotSwarmSimulator](https://github.com/Connor-Mattson/RobotSwarmSimulator) on the [`main` branch](https://github.com/kenblu24/RobotSwarmSimulator/tree/main). I have it installed as **editable** in the venv (`pip install -e RobotSwarmSimulator`)

Most experiments use a common base and utilities from `common/` so make sure that folder is either in the same directory as the experiment script, or somehow on your sys.path .

Common experiment cmd line args
---
Most experiment scripts use a common base. Here are the common command-line args.
Note that the parser is inherited by the scripts, meaning that individual scripts may add, modify, or delete some options.

The first argument is the action to take, which is required.
The three default ones are `run`, `train`, and `test`.
Some options are allowed for all actions:

### common options for `run`, `train`, `test`
* `project`: Specify a project name or path. Networks and logs will be saved to this folder.
    If a path is specified (i.e. contains '/'), it will be used as the project path.
    If a name is specified, the project path will be {root}/{project_name}.
    by default, root is 'out' and project_name is the current time.
    ```cmd
    experiment_?????.py {run|train|test} project_name_here
    ```
* `--root`: Default path to directory to place project folder in
    ```cmd
    experiment_?????.py {run|train|test} --root /path/  # default './out'
    ```
* `-N`, `--agents`: Override the number of agents via the command line:
    ```cmd
    experiment_?????.py {run|train|test} -N 10  # default 10
    ```
* `--cycles`: Override the number of ticks to run the simulation.  
    ```cmd
    experiment_?????.py {run|train|test} -cycles 1000  # default 1000
    ```

### common options for `run`, `test`
To run a single simulation and get the fitness, use
```cmd
experiment_?????.py run
```
To run an experiment-specific test i.e. multiple starting configurations, use
```cmd
experiment_?????.py test --test_args ...
```
Run has several options:
* `--network path/to/network.json`: path to network file. `default="networks/experiment_tenn??????.json"`
* `--noviz`: Explicitly disable viz. By default, viz is on for `test` and `run`.
* `--all_counts_stream '{}'`: A json string indicating where to send network info for visualization.  
    ```cmd
    experiment_?????.py run --all_counts_stream '{"source":"serve","port":8100}'
    ```
* `--stdin`: Tennlab will listen to stdin. Legacy.
* `--prompt`: Wait for a return to continue at each step. Legacy.
* `--viz_delay 0`: Delay between timesteps for viz. `default=None`
* `--viz`: Specify a specific visualizer. `default=True`

### common options for `train`
To start a training session, use
```cmd
experiment_?????.py train
```
Train has several options:
* `-p`: ', type=int, default=1,
                            help="number of threads for concurrent fitness evaluation.")
* `--network path/to/network.json`: output SNN network file path. Will be *overwritten* if it already exists.  
    Default is `networks/experiment_tenn_train.json`, but this is overridden in each experiment script.
* `--logfile path/to/logfile.log`: running log file path. Will be *appended to* if it already exists.  
    Default is `tenn_train.log`, but this is overridden in each experiment script.
* `--label label_name`: label to put into network JSON
* `--save_best_nets`: If specified, the best network from each epoch will be saved to the project folder.
* `--save_all_nets`: If specified, all networks from each epoch will be saved to the project folder.
* `--proc_ticks 10`: Override the number of simulated processor ticks per processor output/action. `default=10`
<!-- * `--population_size 100`: override EONS population size, typically set in the EONS config json. -->
* `--eons_params path/to/eons_config.json`: path to json for eons parameters.
     Default is `eons/std.json`.
* `--snn_params path/to/snn_config.json`: json for processor parameters.
     Default is `config/caspian.json`.
* `--max_fitness 9E9`: Stop EONS early if this fitness is achieved. `default=9999999999`
* `--epochs 999`: Stop EONS after this number of epochs. `default=999`
* `--population_size 100`: Override EONS population size. Defaults to EONS config value.
* `--eons_seed 20`: Seed for EONS. Leave blank for random (time seeded)
* `--graph_distribution path/to/output.tsv`: Specify a file to output fitness distribution over epochs. Will be *overwritten* if it already exists.
* `--viz`: specify a specific visualizer `default=False`


## Commands specific to `experiment_tenn2.py`
These override any of the defaults shown above.

### tenn2 options for `run`, `train`, `test`:
* `--agent_yaml`: path to yaml config for agent  
    Default path is `rss/turbopi-milling/flockbot.yaml`  
* `--world_yaml`: path to yaml config for world  
    Default path is `rss/turbopi-milling/world.yaml`  
\^ **NB:** These two assume that RobotSwarmSimulator is in the directory above this one.

### tenn2 options for `run`: 
* `--track_history`: enable sensor vs. output plotting by clicking on an agent in RobotSwarmSimulator.
* `--log_trajectories`: log sensor vs. output to file.
* `--start_paused`: pause the simulation at startup. Press Space to unpause.

### tenn2 options for `train`:
* `--label label_name`: label to put into network JSON

### tenn2 options for `test`:
`--positions path/to/positions.xlsx`: file containing agent positions. `default=None`  
<!-- `-p`, '--processes': number of threads for concurrent fitness evaluation. `default=1` -->


### tenn2 example commands:

* **Run** a single simulation where each agent is given an SNN network, but don't show a simulation window, just output the final fitness:
    ```cmd
    python experiment_tenn2.py run --network networks/experiment_tenn2_mill20240422_1000t_n10_p100_e1000_s23.json --noviz
    ```  
    Note: By default, the sim will have *10 agents* and stop after *1000 ticks in RSS*.

* **Run** a single simulation, show the sim window, and also have the selected robot in RSS output its SNN state on port 8100:
    ```cmd
    python experiment_tenn2.py run --network networks/experiment_tenn2_mill20240422_1000t_n10_p100_e1000_s23.json --all_counts_stream '{"source":"serve","port":8100}'
    ```  
    Note: you first need to open a visualizer in a separate terminal, like so: 
    ```cmd
    cd framework/viz
    love . -n ../../neuromorphic_experiments/networks/experiment_tenn2_mill20240422_1000t_n10_p100_e1000_s23.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}' --remove_unnecessary_neuron
    ```  
    For more information on the available viz options, see here: https://bitbucket.org/neuromorphic-utk/framework/src/master/viz/


* Start a **training** session across **24 threads** where each sim lasts for **1000 ticks** and has **10 agents**, where the **EONS population size is 100** and is capped at **1000 epochs** and is seeded with the number 20. Save the best network to `results/experiment_tenn2_mill20240627_1000t_n10_p100_e1000_s20.json` and save a tsv with the population fitness data for each epoch to `results/20240627_1000t_n10_p100_e1000.tsv`: 
    ```cmd
    python experiment_tenn2.py train -p 24 --sim_t 1000 --N 10 --pop 100 --epochs 1000 --eons_seed 20 --network results/experiment_tenn2_mill20240627_1000t_n10_p100_e1000_s20.json --graph results/20240627_1000t_n10_p100_e1000.tsv
    ```  
    Note that the best network is overwritten; only one network.json will exist at the end of the session. Also, file names are not checked in any way; you're free to name it whatever you want.

* Start a **testing** session (single-threaded) which evaluates the fitness across multiple pre-generated starting configs:
    ```cmd
    python experiment_tenn2.py test -N 10 --pos test_data.xlsx --net networks/experiment_tenn2_mill20240422_1000t_n10_p100_e1000_s23.json
    ```

* Generate starting configs with pre-set seeds:  
    Open a python REPL i.e. `python`
    ```python
    import rss2
    path = '../RobotSwarmSimulator/demo/configs/flockbots-icra-milling/world.yaml'
    sets = rss2.generate_position_sets(path, num_agents=20, seeds=range(2020, 2025), )
    rss2.save_position_sets_to_xlsx(sets, 'test_positions.xlsx')
    ```  
    Note: You can generate sets for more agents than you test with; 


### Cluster Installation

Installing to the Hopper cluster requires a very specific setup.  
Here's a script to automate the process:  
```console
git clone https://gitlab.orc.gmu.edu/kzhu4/neuromorphic_experiments.git ~/neuromorphic/turtwig
~/neuromorphic/turtwig/scripts/hopper/deploy.sh
```
This will create the environment necessary to run the simulations,
including the dependencies necessary to build Python 3.12.
You will need to add your cluster SSH key to your bitbucket/ORNL account to access the restricted Tennlab framework repository.

Any time you login to the Hopper cluster, you'll need to activate the environment with
```console
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
```

&nbsp;

&nbsp;

&nbsp;

random junk commands
---

<!-- `python experiment_tenn2.py train --processes 16 --sim_time 1000 --agents 10 --training_network networks/experiment_tenn2_mill36w23_1000t_n10.json ` -->
python experiment_tenn2.py train -N 10 --sim_time 1000 --processes 24 --network networks/experiment_tenn2_mill2023w37_1000t_n10_p1000_s20.json --pop 100 --epochs 100 --eons_seed 20 --graph_distribution 1000t_n10_p100_s20.tsv

python experiment_tenn2.py train -N 10 --sim_time 1000 --processes 24 --network networks/experiment_tenn2_mill20240215_0958_1000t_n10_p1000_s20.json --pop 100 --epochs 9999 --eons_seed 20 --graph_distribution 20240215_1000t_n10_p100_s20.tsv



python experiment_tenn2.py run --sim_time 1000 --agents 10 --network results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --all_counts_stream '{"source":"serve","port":8100}'

python experiment_tenn3.py train --sim_time 1000 --agents 10 --network networks/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e100.json --pop 100 --epochs 100 --eons_seed 20 --graph_distribution zespol1000t_n10_p100_s20.tsv

rem visualizer command example
love . -n ../../turtwig/networks/experiment_tenn1_gs34w23_3000t_n20.json --show_input_id --show_output_id -i '{"source":"request","port":"8100", "host": "localhost"}'


love . -n ../../turtwig/results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --show_input_id --show_output_id --remove_unnecessary_neuron -i '{"source":"request","port":"8100", "host": "localhost"}'

python experiment_tenn2.py run --sim_time 1000 --agents 10 --network results/p100/experiment_tenn2_mill2023w38_1000t_n10_p100_s20_e200.json --all_counts_stream '{"source":"serve","port":8100}'

rem
python experiment_tenn2.py train -N 10 --sim_t 1000 -p 24 --pop 100 --epochs 1000 --eons_seed 20 --net results/experiment_tenn2_mill20240417_1000t_n10_p100_e1000_s20.json --graph results/20240417_1000t_n10_p100_e1000_s20.tsv

python experiment_tenn2.py test -N 10 --sim_t 1000 --pos test_data.xlsx --net networks/experiment_tenn2_mill20240422_1000t_n10_p100_e1000_s23.json

python experiment_tenn5.py train -p 24 --sim_t 1000 -N 10 --pop 100 --epochs 1000 --eons_seed 20 --net networks/experiment_tenn5_gs20240529_1000t_n10_p100_e1000_s20.json --graph networks/20240529_1000t_n10_p100_e1000_s20.tsv
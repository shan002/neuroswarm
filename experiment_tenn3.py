# from multiprocessing import Pool, TimeoutError
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import random
import argparse
import os
import time
import pathlib
# import matplotlib.pyplot as plt

# Provided Python utilities from tennlab framework/examples/common
from common.evolver import Evolver
from common.evolver import MPEvolver
import common.utils as nutils
from common.utils import json
from common.application import Application

from zss.flockbot_caspian import FlockbotCaspian

directory = os.path.dirname(os.path.realpath(__file__))


class CustomPool():
    """pool class for Evolver, so we can use tqdm for those sweet progress bars"""

    def __init__(self, **tqdm_kwargs):  # max_workers=args.threads
        self.kwargs = tqdm_kwargs

    def map(self, fn, *iterables):
        return process_map(fn, *iterables, **self.kwargs)


class TennBots(Application):
    """Tennbots application for TennLab neuro framework & Shay Zespol


    """

    def __init__(self, **kwargs):
        self.n_inputs = 2  # default. set one later in the code if necessary
        self.n_outputs = 2  # default. set one later in the code if necessary

        self.env_name = kwargs['environment']
        self.label = kwargs['label']
        self.seed = kwargs['seed']
        self.viz = kwargs['viz']
        self.agents = kwargs['agents']
        self.prompt = kwargs['prompt']
        self.viz_delay = kwargs['viz_delay']
        self.big_fitness = (kwargs['methodology'] == 'big_run')
        self.one_small = (kwargs['methodology'] == 'one_small')
        self.episodes = kwargs['episodes']
        self.processes = kwargs['processes']
        self.sim_time = kwargs['sim_time']
        self.logfile = kwargs['logfile']
        self.run_info = None

        # And set parameters specific to usage.  With testing,
        # we'll read the app_params from the network.  Otherwise, we get them from
        # the command line or defaults.

        app_params = ['proc_ticks', ]
        self.app_params = dict()

        # Copy network from file into memory, including processor & app params
        if kwargs['action'] == "test":
            with open(kwargs['network']) as f:
                j = json.loads(f.read())
            self.net = neuro.Network()
            self.net.from_json(j)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application").to_python()

        # Get params from defaults/cmd params and default proc/eons cfg
        elif kwargs['action'] == "train":
            # self.input_time = kwargs['input_time']
            self.proc_ticks = kwargs['proc_ticks']
            self.training_network = kwargs["training_network"]
            self.eons_params = nutils.load_json_string_file(kwargs["eons_params"])
            self.processor_params = nutils.load_json_string_file(kwargs['processor_params'])
            self.runs = kwargs['runs']

        if kwargs["all_counts_stream"] is not None:
            self.iostream = neuro.IO_Stream()
            j = json.loads(kwargs["all_counts_stream"])
            self.iostream.create_output_from_json(j)
        else:
            self.iostream = None

        # If 'test' and an app param hasn't been set, we simply use defaults (we won't be
        # able to read them from the command line).  This is how we can add information to
        # networks, and not have old networks be obsolete.

        for arg in app_params:
            if not (arg in self.app_params):
                self.app_params[arg] = kwargs[arg]
        # self.flip_up_stay = self.app_params['flip_up_stay']

        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # Get the number of neurons from the agent definition

        self.n_inputs, self.n_outputs, _, _ = FlockbotCaspian.get_default_encoders(10)

        # Set up the initial gym, and set up the action space.

        # env = gym.make(self.env_name)

        # self.action_min = [0]
        # self.action_max = [env.action_space.n-1]
        # self.action_type = env.action_space.dtype.type

        self.log("initialized experiment_tenn3")

        # If we're playing from stdin, we can return now -- nothing
        # else is being used

        if kwargs['action'] == "stdin":
            return

    def fitness(self, processor, network):
        metrics = self.run(processor, network)
        return metrics["Circliness"]

    def run(self, processor, network):
        import zss
        # setup sim

        network.set_data("processor", self.processor_params)

        zespol_config = {
            "num_agents": self.agents,
            "ticks": self.sim_time,
            "ticks_per_second": 7.5,
            "viz": self.viz,
            "network": network,
            "neuro_tpc": self.app_params['proc_ticks']
        }

        world = zss.setup_world(zespol_config)

        for t in range(self.sim_time):
            world.tick()
            metrics = world.metric_window[-1]

        # print(f"final count: {get_how_many_on_goal(world)}")
        # self.run_info = reward_history
        # return reward_history[-1]
        return metrics

    def save_network(self, net, safe_overwrite=True):
        path = pathlib.Path(self.training_network)
        if self.label != "":
            net.set_data("label", self.label)
        net.set_data("processor", self.processor_params)
        net.set_data("application", self.app_params)

        if safe_overwrite and path.is_file():
            path.rename(path.with_suffix('.bak'))
        with open(path, 'w') as f:
            f.write(str(net))

        self.log(f"wrote best network to {str(path)}.")

    def log_status(self, info):
        print(info)
        self.log(info)

    def log(self, msg, timestamp="%Y%m%d %H:%M:%S", prompt=' >>> ', end='\n'):
        if not self.logfile:
            return
        if isinstance(timestamp, str) and '%' in timestamp:
            timestamp: str = time.strftime(timestamp)
        with open(self.logfile, 'a') as f:
            f.write(f"{timestamp}{prompt}{msg}{end}")


def train(**kwargs):
    app = TennBots(**kwargs)

    processes = kwargs["processes"]
    epochs = kwargs["epochs"]
    max_fitness = kwargs["max_fitness"]

    app.log("initialized experiment_tenn3 for training.")

    if processes == 1:
        evolve = Evolver(
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
        )
    else:
        evolve = MPEvolver(  # multi-process for concurrent simulations
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
            pool=CustomPool(max_workers=processes),
        )
    evolve.print_callback = app.log_status

    try:
        evolve.train(epochs, max_fitness)
    except KeyboardInterrupt:
        app.log("training cancelled.")
        raise
    app.log("training finished.")

    # evolve.graph_fitness()


def run(**kwargs):
    app = TennBots(**kwargs)

    # Set up simulator and network

    if kwargs["action"] == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)
        net = app.net

    # Run app and print fitness

    print("Fitness: {:8.4f}".format(app.fitness(proc, net)))


def main():
    # parse cmd line args and run either `train(...)` or `run(...)`

    parser = argparse.ArgumentParser(description='Freeway app for eons, neuro or stdin')
    parser.add_argument('action', choices=['train', 'test', 'stdin'])

    # Parameters that apply to all situations.  These are the only ones that I give defaults.

    parser.add_argument('--seed', default=0, type=int, help="[all] rng seed for the app (0)")
    parser.add_argument('--methodology',
                        choices=['big_run', 'small_runs', 'one_small'], default='big_run',
                        help="[all] one big run/10 small runs/1 small run (big_run).")
    parser.add_argument('--episodes', default=1, type=int, help="[all] # of episodes to run (1)")
    parser.add_argument('--agents', default=10, type=int, help="[all] # of agents to run with.")
    parser.add_argument('--show_collisions',
                        help="[all] print whether there is a collision (unset)",
                        action="store_true")
    parser.add_argument('--show_observations', help="[all] print all 128 observations (unset)",
                        action="store_true")
    parser.add_argument('--show_changes', help="[all] print the observations that change (unset)",
                        action="store_true")
    parser.add_argument('--sim_time', type=int, default=1000,
                        help="[train] time steps per simulate() (9999).")

    # Parameters that only apply to test or stdin.
    # Don't use defaults here, because we don't want the user to specify them when they don't apply.

    parser.add_argument('--viz', help="[test,stdin] use game visualizer", action="store_true")
    parser.add_argument('--noviz', help="[test,stdin] use game visualizer", action="store_true")
    parser.add_argument('--viz_delay', type=float,
                        help="[test,stdin] change the delay between timesteps in the viz (0.016)")
    parser.add_argument('--all_counts_stream', help="[run,stdin] use game visualizer")
    parser.add_argument('--prompt', help="[test] wait for a return to continue at each step.", action="store_true")
    parser.add_argument('--network', help="[test] network file (networks/experiment_tenn3.json)")

    # Parameters that only apply to training - observations and spike encoders.
    # Don't use defaults here, because we don't want the user to specify them when they don't apply.

    parser.add_argument('--label',
                        help="[train] label to put into network JSON (key = label)")
    parser.add_argument('--training_network',
                        help="[train] output network file (networks/experiment_tenn3_train.json)")
    parser.add_argument('--logfile',
                        help="[train] running log file (tenn1_train.log)")

    parser.add_argument('--bias', help="[train] add a bias neuron (unset)", action="store_true")
    parser.add_argument('--flip_up_stay', help="[train] flip output neurons 0 and 1 (unset)",
                        action="store_true")

    parser.add_argument('--decoder',
                        help="[train] json for the SpikeDecoder for player's actions (wta-3)")

    parser.add_argument('--testing_data',
                        help="[test] testing dataset")

    # Parameters that only apply to training - all of the other stuff.
    # Again, don't use defaults, because we don't want the user to specify
    # them when they don't apply.

    parser.add_argument('--proc_ticks', type=float,
                        help="[train] time steps per processor output (10).")
    # parser.add_argument('--input_time', type=float,
    #                     help="[train] time steps over which to pulse input (50).")
    parser.add_argument('--eons_params', help="[train] json for eons parameters (eons/std.json)")
    parser.add_argument('--processor_params',
                        help="[train] json for processor parameters (config/caspian.json)")
    parser.add_argument('--processes', type=int, help="[train] # threads (1)")
    parser.add_argument('--max_fitness', required=False, type=int,
                        help="[train] stop eons if this fitness is achieved (34/1)")
    parser.add_argument('--epochs', required=False, type=int,
                        help="[train] training epochs (50)")
    parser.add_argument('--graph', help="[train] graph eons results", action="store_true")

    # applies to both train, test
    parser.add_argument('--runs', required=False, type=int,
                        help="[train, test] how many runs are used to calculate fitness for a network")

    args = parser.parse_args()

    config = vars(args)

    # Go through the pain of error checking the command line, so that users don't think that they
    # are setting parameters that they are not.  Also set defaults here.

    if args.action == "train":
        illegal = ['viz', 'viz_delay', 'network', 'prompt']
        for s in illegal:
            if config[s]:
                print(f"Cannot specify --{s} with action = {args.action}")
                return
        # if not config['input_time']:
        #     config['input_time'] = 30
        if not config['max_fitness']:
            config['max_fitness'] = 999999999 if config['methodology'] == "big_run" else 1
        if not config['epochs']:
            config['epochs'] = 1000
        if not config['eons_params']:
            config["eons_params"] = os.path.join(directory, 'eons', 'std.json')
        if not config['processor_params']:
            config["processor_params"] = os.path.join(directory, 'config', 'caspian.json')
        if not config['processes']:
            config['processes'] = 4
        if not config['training_network']:
            config["training_network"] = os.path.join(directory, 'networks', 'experiment_tenn3_train.json')
        if not config['logfile']:
            config["logfile"] = "tenn3_train.log"
        if not config['label']:
            config["label"] = ""
        if not config['proc_ticks']:
            config["proc_ticks"] = 10
    else:
        illegal = [
            'eons_params', 'processor_params', 'processes', 'proc_ticks',
            'epochs', 'training_network', 'max_fitness', 'graph',
        ]
        for s in illegal:
            if config[s]:
                print(f"Cannot specify --{s} with action = {args.action}")
                return
        if config['noviz']:
            config['viz'] = False
        else:
            config['viz'] = True
        if not config['viz_delay']:
            config['viz_delay'] = 0.016
        if args.action == "test" or args.action == "run":
            if not config['network']:
                config["network"] = os.path.join(directory, 'networks', 'experiment_tenn3.json')
        if args.action == "test":
            if not config['testing_data']:
                config["testing_data"] = "validation.csv"
        if args.action != "run":
            if not config['all_counts_stream']:
                config["all_counts_stream"] = None
    if args.action in ("train", "test"):
        if not config['runs']:
            config["runs"] = 1

    config["environment"] = "tennbots-v00"

    # Do the appropriate action
    if args.action == "train":
        train(**config)
    else:
        run(**config)


if __name__ == "__main__":
    main()

# TODO: spike encoders (perhaps bins with rate encoding)

# from multiprocessing import Pool, TimeoutError
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import random
import argparse
import os
import time
import pathlib
import inspect
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

    def __init__(self, args):
        self.env_name = args.environment
        self.label = args.label
        self.seed = args.seed
        self.viz = args.viz
        self.agents = args.agents
        self.prompt = args.prompt
        self.viz_delay = args.viz_delay
        self.processes = args.processes
        self.sim_time = args.sim_time
        self.logfile = args.logfile
        self.run_info = None
        # And set parameters specific to usage.  With testing,
        # we'll read the app_params from the network.  Otherwise, we get them from
        # the command line or defaults.

        app_params = ['proc_ticks', ]
        self.app_params = dict()

        # Copy network from file into memory, including processor & app params
        if args.action == "test":
            with open(args.network) as f:
                j = json.loads(f.read())
            self.net = neuro.Network()
            self.net.from_json(j)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application").to_python()

        # Get params from defaults/cmd params and default proc/eons cfg
        elif args.action == "train":
            # self.input_time = args.input_time
            self.proc_ticks = args.proc_ticks
            self.training_network = args.training_network
            self.eons_params = nutils.load_json_string_file(args.eons_params)
            self.processor_params = nutils.load_json_string_file(args.processor_params)
            self.runs = args.runs

        if args.all_counts_stream is not None:
            self.iostream = neuro.IO_Stream()
            j = json.loads(args.all_counts_stream)
            self.iostream.create_output_from_json(j)
        else:
            self.iostream = None

        # If an app param hasn't been set on the network, use defaults.
        # This helps prevent old networks from becoming obsolete.
        for arg in app_params:
            if not (arg in self.app_params):
                self.app_params[arg] = vars(args)[arg]

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


def train(args):
    app = TennBots(args)

    processes = args.processes
    epochs = args.epochs
    max_fitness = args.max_fitness

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


def run(args):
    app = TennBots(args)

    # Set up simulator and network

    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)
        net = app.net

    # Run app and print fitness

    print("Fitness: {:8.4f}".format(app.fitness(proc, net)))


def get_parser():
    # parse cmd line args and run either `train(...)` or `run(...)`
    HelpDefaults = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description='Script for running Zespol sims for milling.',
        formatter_class=HelpDefaults,
    )

    subpar = parser.add_subparsers(required=True, dest='action', metavar='ACTION')
    sub_train = subpar.add_parser('train', help='Do training using EONS', formatter_class=HelpDefaults)
    sub_test = subpar.add_parser('test', help='Validate over a testing set and output the score.',
                                 aliases=['validate'], formatter_class=HelpDefaults)
    sub_run = subpar.add_parser('run', help='Run a simulation and output the score.', formatter_class=HelpDefaults)

    # Parameters that apply to all situations.
    parser.add_argument('--seed', type=int, help="rng seed for the app", default=0)
    parser.add_argument('-N', '--agents', type=int, help="# of agents to run with.", default=10)
    parser.add_argument('--sim_time', type=int, default=1000,
                        help="time steps per simulate.")

    for sub in (sub_test, sub_run):  # arguments that apply to test/validation and stdin
        sub.add_argument('--stdin', help="Use stdin as input.", action="store_true")
        sub.add_argument('--prompt', help="wait for a return to continue at each step.", action="store_true")
        sub.add_argument('--network', help="network", default="networks/experiment_tenn3.json")
        sub.add_argument('--viz', type=str, help="specify a specific visualizer")
        sub.add_argument('--noviz', help="explicitly disable viz", action="store_true")
        sub.add_argument('--viz_delay', type=float,  # default: None
                         help="delay between timesteps for viz.")
        sub.add_argument('--all_counts_stream', type=str, help="""
            Takes a json string.
            If supplied, this will enable sending of network
            info to iostream for network visualization.
            e.g. '{"source":"serve","port":8100}'
        """)

    # Training args
    sub_train.add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sub_train.add_argument('--network', default="networks/experiment_tenn3_train.json",
                           help="output network file path.")
    sub_train.add_argument('--logfile', default="tenn1_train.log",
                           help="running log file path.")

    sub_train.add_argument('--proc_ticks', type=int, default=10,
                           help="time steps per processor output.")
    # sub_train.add_argument('--input_time', type=float,
    #                     help="[train] time steps over which to pulse input (50).")
    sub_train.add_argument('--eons_params', default="eons/std.json",
                           help="json for eons parameters.")
    sub_train.add_argument('--processor_params', default="config/caspian.json",
                           help="json for processor parameters.")
    sub_train.add_argument('-p', '--processes', type=int, default=1,
                           help="number of threads for concurrent fitness evaluation.")
    sub_train.add_argument('--max_fitness', type=float, default=9999999999,
                           help="stop eons if this fitness is achieved.")
    sub_train.add_argument('--epochs', type=int, default=999,
                           help="training epochs")
    sub_train.add_argument('--graph_distribution', help="Specify a file to output fitness distribution over epochs.")

    for sub in (sub_train, sub_test):  # applies to both train, test
        sub.add_argument('--runs', type=int, default=1,
                         help="how many runs are used to calculate fitness for a network.")

    sub_test.add_argument('--testing_data', required=True,
                          help="testing dataset file path.")

    return parser


def main(args):
    args.environment = "tennbots-v00"

    # Do the appropriate action
    if args.action == "train":
        train(args)
    else:
        if args.noviz:
            args.viz = False
        run(args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

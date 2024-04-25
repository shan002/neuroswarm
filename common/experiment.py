from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import os
import time
import pathlib
# import matplotlib.pyplot as plt

# import custom cmd parser
import util.argparse

# Provided Python utilities from tennlab framework/examples/common
from .evolver import Evolver
from .evolver import MPEvolver
from . import jsontools as jst
from .application import Application


class CustomPool():
    """pool class for Evolver, so we can use tqdm for those sweet progress bars"""

    def __init__(self, **tqdm_kwargs):  # type:ignore[reportMissingSuperCall] # max_workers=args.threads
        self.kwargs = tqdm_kwargs

    def map(self, fn, *iterables):
        return process_map(fn, *iterables, **self.kwargs)


class TennExperiment(Application):

    def __init__(self, args):
        super().__init__()
        self.env_name = args.environment
        self.label = args.label
        # self.eons_seed = args.eons_seed
        self.viz = args.viz
        self.noviz = args.noviz
        self.agents = args.agents
        self.prompt = args.prompt
        self.viz_delay = args.viz_delay
        self.processes = args.processes
        self.sim_time = args.sim_time
        self.logfile = args.logfile
        self.graph_distribution = args.graph_distribution

        # And set parameters specific to usage.  With testing,
        # we'll read the app_params from the network.  Otherwise, we get them from
        # the command line or defaults.

        app_params = ['proc_ticks', ]
        self.app_params = dict()

        # Copy network from file into memory, including processor & app params
        if args.action in ["test", "run", "validate"]:
            j = jst.smartload(args.network)
            self.net = neuro.Network()
            self.net.from_json(j)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application").to_python()

        # Get params from defaults/cmd params and default proc/eons cfg
        elif args.action == "train":
            # self.input_time = args.input_time
            self.proc_ticks = args.proc_ticks
            self.training_network = args.network
            self.save_multiple = args.save_best_nets
            self.eons_params = jst.smartload(args.eons_params)
            self.processor_params = jst.smartload(args.processor_params)
            self.runs = args.runs
            self.check_output_path()

        if args.all_counts_stream is not None:
            self.iostream = neuro.IO_Stream()
            j = jst.smartload(args.all_counts_stream)
            self.iostream.create_output_from_json(j)
        else:
            self.iostream = None

        # If an app param hasn't been set on the network, use defaults.
        # This helps prevent old networks from becoming obsolete.
        for arg in app_params:
            if arg not in self.app_params:
                self.app_params[arg] = vars(args)[arg]

        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # Get the number of neurons from the agent definition

        self.n_inputs, self.n_outputs, = None, None

        if self.graph_distribution:
            self.log(str(args), logpath=self.graph_distribution)
            with open(self.graph_distribution, 'a') as f:
                f.write(f"{time.time()}\t{0}\t[]\n")

    def run(self, processor, network):
        return None

    def check_output_path(self):
        path = pathlib.Path(self.training_network)

        def check_if_writable(path):
            if not os.access(path, os.W_OK):
                raise PermissionError(
                    f"{path} could not be accessed. Check that you have permissions to write to it."
                )
        if self.save_multiple:
            if not path.is_dir():
                # raise OSError(1, f"{path} is a directory. Please specify a valid path for the output network file.")
                input(f"Destination directory not found.\nCTRL+C to cancel or Press enter to create:\n\t{str(path)}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                input(f"Files in destination directory may be overwritten:\n\t{str(path)}")
        else:
            if path.is_file():
                check_if_writable(path)
                print(f"WARNING: The output network file\n    {path}\nexists and will be overwritten!")
            elif path.is_dir():
                raise OSError(1, f"{path} is a directory. Please specify a valid path for the output network file.")
            else:  # parent dir probably doesn't exist.
                if path.parent.is_dir():
                    try:
                        f = open(path, 'ab')
                    except BaseException as err:
                        raise err
                    finally:
                        f.close()
                else:
                    raise OSError(2, f"One or more parent directories are missing. Cannot write to {path}.")

    def save_network(self, info, newchamp=True, safe_overwrite=True):
        if not self.save_multiple and not newchamp:
            return

        net = info.best_network
        path = pathlib.Path(self.training_network)
        if self.label != "":
            net.set_data("label", self.label)
        net.set_data("processor", self.processor_params)
        net.set_data("application", self.app_params)

        if self.save_multiple:
            path /= f"{info.i}.json"
            safe_overwrite = False

        if safe_overwrite and path.is_file():
            path.rename(path.with_suffix('.bak'))
        with open(path, 'w') as f:
            f.write(str(net))

        self.log(f"wrote best network to {str(path)}.")

    def log_status(self, info):
        print(info)
        self.log(info)

        if self.graph_distribution:
            with open(self.graph_distribution, 'a') as f:
                f.write(f"{time.time()}\t{info.i}\t{repr(info.fitnesses)}\n")

    def log(self, msg, timestamp="%Y%m%d %H:%M:%S", prompt=' >>> ', end='\n', logpath=None):
        if logpath is None:
            logpath = self.logfile
        if logpath is None:
            return
        if isinstance(timestamp, str) and '%' in timestamp:
            timestamp: str = time.strftime(timestamp)
        with open(logpath, 'a') as f:
            f.write(f"{timestamp}{prompt}{msg}{end}")


def train(app, args):

    processes = args.processes
    epochs = args.epochs
    max_fitness = args.max_fitness

    if args.population_size is not None:  # if specified, override eons_params file
        app.eons_params["population_size"] = args.population_size
    if args.eons_seed is not None:  # if specified, force EONS seed
        app.eons_params["seed_eo"] = args.eons_seed

    app.log(f"initialized {args.environment} for training.")

    if processes == 1:
        evolve = Evolver(
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
        )
        evolve.net_callback = lambda x: tqdm(x,)
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
        return evolve.train(epochs, max_fitness)
    except KeyboardInterrupt:
        app.log("training cancelled.")
        raise
    finally:
        app.log("training finished.")

    # evolve.graph_fitness()


def run(app, args):

    # Set up simulator and network

    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)
        net = app.net

    # Run app and print fitness
    fitness = app.fitness(proc, net)
    print(f"Fitness: {fitness:8.4f}")
    return fitness


def get_parsers(conflict_handler='resolve'):
    # parse cmd line args and run either `train(...)` or `run(...)`
    HelpDefaults = util.argparse.ArgumentDefaultsHelpFormatter
    ch = conflict_handler
    parser = util.argparse.ArgumentParser(
        description='Script for running an experiment or training an SNN with EONS.',
        formatter_class=HelpDefaults,
        conflict_handler=ch,
    )

    subpar = parser.add_subparsers(required=True, dest='action', metavar='ACTION', add_all=True)
    sub_train = subpar.add_parser('train', help='Do training using EONS', formatter_class=HelpDefaults, conflict_handler=ch,)
    sub_test = subpar.add_parser('test', help='Validate over a testing set and output the score.',
                                 aliases=['validate'], formatter_class=HelpDefaults, conflict_handler=ch,)
    sub_run = subpar.add_parser('run', help='Run a simulation and output the score.',
                                formatter_class=HelpDefaults, conflict_handler=ch,)

    for sub in subpar.parsers.values():  # applies to everything
        # sub.add_argument('--seed', type=int, help="rng seed for the app", default=None)
        sub.add_argument('-N', '--agents', type=int, help="# of agents to run with.", default=10)
        sub.add_argument('--sim_time', type=int, default=1000,
                         help="time steps per simulate.")

    for sub in (sub_test, sub_run):  # arguments that apply to test/validation and stdin
        sub.add_argument('--stdin', help="Use stdin as input.", action="store_true")
        sub.add_argument('--prompt', help="wait for a return to continue at each step.", action="store_true")
        sub.add_argument('--network', help="network", default="networks/experiment_tenn.json")
        sub.add_argument('--viz', help="specify a specific visualizer", default=True)
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
    sub_train.add_argument('--network', default="networks/experiment_tenn_train.json",
                           help="output network file path.")
    sub_train.add_argument('--save_best_nets', action='store_true')
    sub_train.add_argument('--logfile', default="tenn_train.log",
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
    sub_train.add_argument('--population_size', type=int,
                           help="override population size")
    sub_train.add_argument('--eons_seed', type=int,
                           help="Seed for EONS. Leave blank for random (time)")
    sub_train.add_argument('--graph_distribution', help="Specify a file to output fitness distribution over epochs.")
    sub_train.add_argument('--viz', help="specify a specific visualizer", default=False)

    for sub in (sub_train, sub_test):  # applies to both train, test
        sub.add_argument('--runs', type=int, default=1,
                         help="how many runs are used to calculate fitness for a network.")

    # sub_test.add_argument('--testing_data', required=True,
    #                       help="testing dataset file path.")

    return parser, subpar


def main(app, args):
    args.environment = "kevin_experiment-v00"

    # Do the appropriate action
    if args.action == "train":
        train(app, args)
    else:
        if args.noviz:
            args.viz = False
        run(app, args)


if __name__ == "__main__":
    parser, subpar = get_parsers()
    args = parser.parse_args()
    main(TennExperiment(args), args)

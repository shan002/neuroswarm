from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import os
import sys
import time
import pathlib
import shutil

# import custom cmd parser
import common.argparse

# Provided Python utilities from tennlab framework/examples/common
from .evolver import Evolver
from .evolver import MPEvolver
from . import jsontools as jst
from .application import Application


DEFAULT_PROJECT_BASEPATH = pathlib.Path("results")
DEFAULT_LOGFILE_NAME = "training.log"
DEFAULT_BESTNET_NAME = "best.json"
BACKUPNET_NAME = "previous.json"
DEFAULT_POPULATION_FITNESS_NAME = "population_fitnesses.log"
NETWORKS_DIR_NAME = "networks"


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
        self.eons_seed = args.eons_seed
        self.viz = args.viz
        self.noviz = args.noviz
        self.agents = args.agents
        self.viz_delay = args.viz_delay
        self.processes = args.processes
        self.cycles = args.cycles

        if args.save_all_nets:
            self.save_strategy = "all"
        elif args.save_best_nets:
            self.save_strategy = "best_per_epoch"
        else:
            self.save_strategy = "one_best"

        # set project mode.
        if args.noproj:
            # don't create a project folder. Some features will be unavailable.
            self.project_path = None
            self.logfile = args.logfile
            self.best_network_path = args.network
        else:
            # Okay, we're doing a project folder.
            # first set the project name
            if args.project is None:
                # no project name specified, so use the experiment name and timestamp
                self.project_name = f"{args.environment}-{time.strftime('%Y%m%d-%H%M%S')}"
                self.project_path = DEFAULT_PROJECT_BASEPATH / pathlib.Path(self.project_name)
            else:
                self.project_name = args.project
                self.project_path = pathlib.Path(self.project_name)
            # now set paths for particular files (and apply cmdline overrides)
            self.logfile = (self.project_path / DEFAULT_LOGFILE_NAME) if args.logfile is None else args.logfile
            self.best_network_path = (self.project_path / DEFAULT_BESTNET_NAME) if args.network is None else args.network
            self.popfit_path = None if args.dont_log_population_fitnesses else (
                self.project_path / "population_fitnesses.log")

        app_params = ['encoder_ticks', ]
        self.app_params = dict()

        if args.action in ["test", "run", "validate"]:
            j = jst.smartload(args.network)
            self.net = neuro.Network()
            self.net.from_json(j)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application")
            # TODO: inquirer.py network chooser if no network specified

        # Get params from defaults/cmd params and default proc/eons cfg
        elif args.action == "train":
            self.encoder_ticks = args.encoder_ticks
            self.eons_params = jst.smartload(args.eons_params)
            self.processor_params = jst.smartload(args.snn_params)
            self.runs = args.runs

            if self.project_path:
                # check if the project folder exists
                if self.project_path.is_dir():
                    s = input(f"Project folder already exists:\n\t{str(self.project_path)}\n'y' to continue, 'rm' to delete the contents of the folder, anything else to exit.")  # noqa: E501
                    if s.lower() != 'y':
                        sys.exit(1)
                    if s.lower() == 'rm':
                        shutil.rmtree(self.project_path)
                # if not, create it
                else:
                    self.project_path.mkdir(parents=True, exist_ok=True)
            if self.popfit_path:
                self.log(str(args), logpath=self.popfit_path)
                with open(self.popfit_path, 'a') as f:
                    f.write(f"{time.time()}\t{0}\t[]\n")
            self.check_output_path()
            self.max_epochs: int

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

        self.n_inputs, self.n_outputs, = 0, 0

    def run(self, processor, network):
        return None

    def check_output_path(self):
        # check that the best network path is writable
        path = pathlib.Path(self.best_network_path)

        def check_if_writable(path):
            if not os.access(path, os.W_OK):
                msg = f"{path} could not be accessed. Check that you have permissions to write to it."
                raise PermissionError(msg)

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
                    f.close()  # type: ignore
            else:
                raise OSError(2, f"One or more parent directories are missing. Cannot write to {path}.")

    def pre_epoch(self, eons):
        if self.save_strategy == "all":
            max_epochs_digits = len(str(self.max_epochs))
            population_digits = len(str(len(eons.pop.networks)))
            (self.project_path / NETWORKS_DIR_NAME).mkdir(parents=False, exist_ok=True)
            for nid, net in enumerate(eons.pop.networks):
                path = self.project_path / NETWORKS_DIR_NAME / f"e{info.i:0{max_epochs_digits}}-{nid:0{population_digits}}.json"
                self.save_network(net.network, path)
        # elif self.save_strategy == "one_best":

    def post_epoch(self, eons, info, new_best):  # eons calls this after each epoch
        if self.save_strategy == "best_per_epoch":
            ndigits = len(str(self.max_epochs))
            path = self.project_path / NETWORKS_DIR_NAME / f"e{info.i:0{ndigits}}.json"
            (self.project_path / NETWORKS_DIR_NAME).mkdir(parents=False, exist_ok=True)
            self.save_network(info.best_network, path)
        # do this regardless of save_strategy
        if new_best:
            self.save_best_network(info, safe_overwrite=True)

        self.log_status(info)  # print and log epoch info
        self.log_fitnesses(info)  # write population fitnesses to file

    def save_network(self, net, path):
        if self.label:
            net.set_data("label", self.label)
        net.set_data("processor", self.processor_params)
        net.set_data("application", self.app_params)
        with open(path, 'w') as f:
            f.write(str(net))

    def save_best_network(self, info, safe_overwrite=True):
        net = info.best_network
        path = self.best_network_path

        if safe_overwrite and path.is_file():
            path.rename(path.with_name(BACKUPNET_NAME))

        self.save_network(net, path)

        self.log(f"wrote best network to {str(path)}.")

    def log_fitnesses(self, info):
        if self.popfit_path:
            with open(self.popfit_path, 'a') as f:
                f.write(f"{time.time()}\t{info.i}\t{repr(info.fitnesses)}\n")

    def log_status(self, info):
        print(info)
        self.log(info)

    def log(self, msg, timestamp="%Y%m%d %H:%M:%S", prompt=' >>> ', end='\n', logpath=None):
        if logpath is None:
            logpath = self.logfile  # try using self.logfile (likely set in init)
        if logpath is None:
            return
        if isinstance(timestamp, str) and '%' in timestamp:
            timestamp: str = time.strftime(timestamp)
        with open(logpath, 'a') as f:
            f.write(f"{timestamp}{prompt}{msg}{end}")


def train(app, args):

    processes = args.processes
    app.max_epochs = args.epochs
    max_fitness = args.max_fitness

    if args.population_size is not None:  # if specified, override eons_params file
        app.eons_params["population_size"] = args.population_size
    if args.eons_seed is not None:  # if specified, force EONS seed
        app.eons_params["seed_eo"] = args.eons_seed

    app.log(f"initialized {args.environment} for training.")

    eons_args = {
            'app': app,
            'eons_params': app.eons_params,
            'proc_name': caspian.Processor,
            'proc_params': app.processor_params,
            'stop_fitness': max_fitness,
            'do_print': False,
    }

    if processes == 1:
        evolve = Evolver(
            **eons_args,
        )
        evolve.net_callback = lambda x: tqdm(x,)  # type: ignore[reportAttributeAccessIssue]
    else:
        evolve = MPEvolver(  # multi-process for concurrent simulations
            **eons_args,
            pool=CustomPool(max_workers=processes),  # type: ignore[reportArgumentType]
        )

    try:
        return evolve.train(app.max_epochs)
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
    HelpDefaults = common.argparse.ArgumentDefaultsHelpFormatter
    ch = conflict_handler
    parser = common.argparse.ArgumentParser(
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
        sub.add_argument('--cycles', type=int, default=1000,
                         help="time steps to simulate.")

    for sub in (sub_test, sub_run):  # arguments that apply to test/validation and stdin
        sub.add_argument('--stdin', help="Use stdin as input.", action="store_true")
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
    savestrat = sub_train.add_mutually_exclusive_group(required=False)
    savestrat.add_argument('--save_best_nets', action='store_true')
    savestrat.add_argument('--save_all_nets', action='store_true')
    proj_mode = sub_train.add_mutually_exclusive_group(required=False)
    proj_mode.add_argument('--project', default=None, help="""
        Specify a project name or path.
        If a path is specified (i.e. contains '/'), it will be used as the project path.
        If a name is specified, the project path will be results/{project_name}.
    """)
    proj_mode.add_argument('--noproj', action='store_true', help="don't create a project folder")
    sub_train.add_argument('--network', default=None,
                           help="Force best network file path. By default, this is saved to the projectdir/best.json")
    sub_train.add_argument('--logfile', default=None,
                           help="running log file path. By default, this is saved to the projectdir/training.log")

    sub_train.add_argument('--encoder_ticks', type=int, default=10,
                           help="Used to determine the encoder/decoder ticks.")
    sub_train.add_argument('--extra_ticks', type=int, default=5,
                        help="Extra ticks to account for propagation time.")
    sub_train.add_argument('--eons_params', default="eons/std.json",
                           help="json for eons parameters.")
    sub_train.add_argument('--snn_params', default="config/caspian.json",
                           help="json for SNN processor parameters.")
    sub_train.add_argument('-p', '--processes', type=int, default=1,
                           help="number of threads for concurrent fitness evaluation.")
    sub_train.add_argument('--max_fitness', type=float, default=9999999999,
                           help="stop eons if this fitness is achieved.")
    sub_train.add_argument('--epochs', type=int, default=999,
                           help="training epochs limit")
    sub_train.add_argument('--population_size', type=int,
                           help="override population size")
    sub_train.add_argument('--eons_seed', type=int,
                           help="Seed for EONS. Leave blank for random (time)")
    sub_train.add_argument('--dont_log_population_fitnesses', action="store_true",
                           help="disable logging of population fitnesses")
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

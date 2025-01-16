from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import os
import sys
import platform
import time
import pathlib
import shutil
import yaml
import re

# import custom cmd parser
import common.argparse

# Provided Python utilities from tennlab framework/examples/common
from .evolver import Evolver
from .evolver import MPEvolver
from . import jsontools as jst
from .application import Application
from . import project
from . import env_tools as envt


RE_CONTAINS_SEP = re.compile(r"[/\\]")
DEFAULT_PROJECT_BASEPATH = pathlib.Path("out")

# class CustomPool():
#     """pool class for Evolver, so we can use tqdm for those sweet progress bars"""

#     def __init__(self, **tqdm_kwargs):  # type:ignore[reportMissingSuperCall] # max_workers=args.threads
#         self.kwargs = tqdm_kwargs

#     def map(self, fn, iterables):
#         breakpoint()
#         return process_map(fn, iterables)


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
        self.p: project.Project | project.FolderlessProject
        if args.project is None and args.network:
            # don't create a project folder. Some features will be unavailable.
            self.p = project.FolderlessProject(args.network, args.logfile)
            self.log_fitnesses = lambda x: None
        else:
            # Okay, we're doing a project folder.
            # first set the project name
            if args.project is None and args.action == "train":
                # no project name specified, so use the experiment name and timestamp
                project_name = f"{time.strftime('%y%m%d-%H%M%S')}-{args.environment}"
                self.p = project.Project(path=args.root / project_name, name=project_name)
            elif args.project is None:
                # no project name specified; ask user
                path = project.inquire_project(root=args.root)
                self.p = project.Project(path=path, name=path.name)
            elif RE_CONTAINS_SEP.search(args.project):  # project name contains a path separator
                project_name = pathlib.Path(args.project).name
                if args.root is not DEFAULT_PROJECT_BASEPATH:
                    print("WARNING: You seem to have specified a root path AND a full project path.")
                    print(f"The root path will be ignored; path={args.project}")
                self.p = project.Project(path=args.project, name=project_name)
            else:
                project_name = args.project
                self.p = project.Project(path=args.root / project_name, name=project_name, overwrite=args.overwrite_project)
            self.log_fitnesses = self.p.log_popfit  # type: ignore[reportAttributeAccessIssue] for if project is FolderlessProject

        # app_params = ['encoder_ticks', ]
        self.app_params = dict()

        if args.action in ["test", "run", "validate"]:
            if args.prnet:
                raise NotImplementedError("TODO: load network from project folder")  # TODO
            self.net = neuro.Network()
            self.p.load_bestnet(args.network)  # defaults to best.json if not specified via args
            self.net.from_json(self.p.bestnet)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application")
            # TODO: inquirer.py network chooser if no network specified

        # Get params from defaults/cmd params and default proc/eons cfg
        elif args.action == "train":
            self.eons_params = jst.smartload(args.eons_params)
            self.processor_params = jst.smartload(args.snn_params)
            self.runs = args.runs
            self.p.make_root_interactive()
            self.p.check_bestnet_writable()

        if args.all_counts_stream is not None:
            self.iostream = neuro.IO_Stream()
            j = jst.smartload(args.all_counts_stream)
            self.iostream.create_output_from_json(j)
        else:
            self.iostream = None

        # If an app param hasn't been set on the network, use defaults.
        # This helps prevent old networks from becoming obsolete.
        # for arg in app_params:
        #     if arg not in self.app_params:
        #         self.app_params[arg] = vars(args)[arg]

        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # Get the number of neurons from the agent definition

        self.n_inputs, self.n_outputs, = 0, 0
        self.args = args

    def run(self, processor, network):
        return None

    def pre_epoch(self, eons):
        if self.save_strategy == "all":
            for nid, net in enumerate(eons.pop.networks):
                self.save_network(net.network, self.p.networks.get_file(eons.i, nid))  # type: ignore[reportAttributeAccessIssue] for if project is FolderlessProject

    def post_epoch(self, eons, info, new_best):  # eons calls this after each epoch
        if self.save_strategy == "best_per_epoch":
            self.save_network(info.best_network, self.p.networks.get_file(info.i, info.best_net_id)) # type: ignore[reportAttributeAccessIssue] for if project is FolderlessProject
        if new_best:  # save best network to its own file regardless of save_strategy
            self.save_best_network(info, safe_overwrite=True)
        self.log_status(info)  # print and log epoch info
        self.log_fitnesses(info)  # write population fitnesses to file

    def save_network(self, net, path):
        if self.label:
            net.set_data("label", self.label)
        net.set_data("processor", self.processor_params)
        net.set_data("application", self.app_params)
        self.p.bestnet = net
        path.write(str(net))

    def save_best_network(self, info, safe_overwrite=True):
        self.p.backup_bestnet()
        self.save_network(info.best_network, self.p.bestnet_file)
        self.log(f"wrote best network to {str(self.p.bestnet_file)}.")

    def log_status(self, info):
        print(info)
        self.log(info)

    def log(self, msg, timestamp="%Y%m%d %H:%M:%S", prompt=' >>> ', end='\n'):
        if isinstance(timestamp, str) and '%' in timestamp:
            timestamp: str = time.strftime(timestamp)
        self.p.logfile += (f"{timestamp}{prompt}{msg}{end}")

    def save_artifacts(self, evolver):
        if not isinstance(self.p, project.Project):
            return
        self.p: project.Project
        self.p.save_yaml_artifact("evolver.yaml", evolver)
        self.p.save_yaml_artifact("experiment.yaml", self)
        return True

    def args_as_dict(self):
        return vars(self.args)

    def as_config_dict(self):
        return {
            "args": self.args_as_dict(),
            "env_info": self.get_env_info(),
            "app_params": self.app_params,
            "label": self.label,
            "eons_seed": self.eons_seed,
            "agents": self.agents,
            "processes": self.processes,
            "cycles": self.cycles,
            "save_strategy": self.save_strategy,
            "p": self.p,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }

    def get_env_info(self):
        d = {
            "uname": platform.uname()._asdict(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
            "cmd": ' '.join(sys.argv),
            ".dependencies": {},
        }
        try:
            d.update({
                "branch": envt.get_branch_name('.'),
                "HEAD": envt.git_hash('.'),
                "status": [s.strip() for s in envt.git_porcelain('.').split('\n')],
            })
        except Exception:
            d.update({"git_repo": None})
        return d

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
            'tqdm': True,
    }

    if processes == 1:
        evolve = Evolver(
            **eons_args,
        )
        # evolve.net_callback = lambda x: tqdm(x,)  # type: ignore[reportAttributeAccessIssue]
        if args.processes is None:
            print(f"Using single detected CPU (single threaded).")
        else:
            print(f"Using single thread.")
    else:
        evolve = MPEvolver(  # multi-process for concurrent simulations
            **eons_args,
            max_workers=processes,  # type: ignore[reportArgumentType]
        )
        if args.processes is None:
            print(f"Using {os.cpu_count()} detected CPUs/threads.")
        else:
            print(f"Using {args.processes} threads.")

    app.save_artifacts(evolve)

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
        sub.add_argument('project', nargs='?', help="""
            Specify a project name or path.
            If a path is specified (i.e. contains '/'), it will be used as the project path.
            If a name is specified, the project path will be {root}/{project_name}.
            by default, root is 'out' and project_name is the current time.
        """)
        sub.add_argument('--root', help="Default path to directory to place project folder in",
                         type=pathlib.Path, default=DEFAULT_PROJECT_BASEPATH)
        # sub.add_argument('--seed', type=int, help="rng seed for the app", default=None)
        sub.add_argument('-N', '--agents', type=int, help="# of agents to run with.", default=10)
        sub.add_argument('--cycles', type=int, default=1000,
                         help="time steps to simulate.")

    for sub in (sub_test, sub_run):  # arguments that apply to test/validation and stdin
        sub.add_argument('--stdin', help="Use stdin as input.", action="store_true")
        net = sub.add_mutually_exclusive_group(required=False)
        net.add_argument('--network', help="network")
        net.add_argument('--prnet', help="network name in the project folder")
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
    sub_train.add_argument('--network', default=None,
                           help="Force best network file path. By default, this is saved to the projectdir/best.json")
    sub_train.add_argument('--logfile', default=None,
                           help="running log file path. By default, this is saved to the projectdir/training.log")
    sub_train.add_argument('--overwrite_project', action='store_true',
                           help="overwrite project folder if it already exists.")
    savestrat = sub_train.add_mutually_exclusive_group(required=False)
    savestrat.add_argument('--save_best_nets', action='store_true')
    savestrat.add_argument('--save_all_nets', action='store_true')

    # sub_train.add_argument('--encoder_ticks', type=int, default=None,
    #                        help="Used to determine the encoder/decoder ticks.")
    # sub_train.add_argument('--extra_ticks', type=int, default=5,
    #                     help="Extra ticks to account for propagation time.")
    sub_train.add_argument('--eons_params', default="eons/std.json",
                           help="json for eons parameters.")
    sub_train.add_argument('--snn_params', default="config/caspian.json",
                           help="json for SNN processor parameters.")
    sub_train.add_argument('-p', '--processes', type=int, default=None,
                           help="number of threads for concurrent fitness evaluation. Defaults to detected CPU count.")
    sub_train.add_argument('--max_fitness', type=float, default=9999999999,
                           help="stop eons if this fitness is achieved.")
    sub_train.add_argument('--epochs', type=int, default=999,
                           help="training epochs limit")
    sub_train.add_argument('--population_size', type=int,
                           help="override population size")
    sub_train.add_argument('--eons_seed', type=int,
                           help="Seed for EONS. Leave blank for random (time)")
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

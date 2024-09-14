from io import BytesIO
import os
# import matplotlib.pyplot as plt

import caspian

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment
from common.utils import make_template
from common import env_tools as envt

from novel_swarms.agent.MillingAgentCaspian import MillingAgentCaspianConfig
from novel_swarms.agent.MillingAgentCaspian import MillingAgentCaspian
from novel_swarms.behavior import Circliness

from rss.gui import TennlabGUI
import rss.graphing as graphing

# typing:
from typing import override

from common.argparse import ArgumentError


class ConnorMillingExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    def __init__(self, args):
        super().__init__(args)
        self.agent_yaml = args.agent_yaml
        self.world_yaml = args.world_yaml
        self.run_info = None

        self.n_inputs, self.n_outputs, _, _ = MillingAgentCaspian.get_default_encoders()

        self.track_history = args.track_history or args.log_trajectories
        self.log_trajectories = args.log_trajectories

        self.start_paused = getattr(args, 'start_paused', False)

        self.log("initialized experiment_tenn2")

    def simulate(self, processor, network, init_callback=lambda x: x):
        import rss.rss2 as rss
        # setup sim

        network.set_data("processor", self.processor_params)

        robot_config = rss.configure_robots(network, MillingAgentCaspianConfig, agent_yaml_path=self.agent_yaml,
                                             track_all=self.viz, track_io=self.track_history)
        world = rss.create_environment(robot_config=robot_config, world_yaml_path=self.world_yaml,
                                        num_agents=self.agents, stop_at=self.cycles)
        world.behavior = [Circliness(history=max(self.cycles, 1), avg_history_max=450)]

        def callback(world, screen):
            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.neuron_ids,
                    "Event Counts": a.neuron_counts
                })

        gui = TennlabGUI(x=0, y=0, h=0, w=300)
        gui.position = "sidebar_right"
        if self.viz is False or self.noviz:
            gui = False

        world_subscriber = rss.WorldSubscriber(func=callback)

        world = init_callback(world)

        world_output = rss.simulator(  # type:ignore[reportPrivateLocalImportUsage]  # run simulator
            world_config=world,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
            start_paused=self.start_paused,
        )
        return world_output

    def extract_fitness(self, world_output):
        self.run_info = world_output.behavior[0].value_history
        return world_output.behavior[0].out_current()[1]

    @override
    def fitness(self, processor, network, init_callback=lambda x: x):
        world_output = self.simulate(processor, network, init_callback)
        return self.extract_fitness(world_output)

    def as_config_dict(self):
        d = super().as_config_dict()
        d.update({
            "agent_yaml_path": self.agent_yaml,
            "world_yaml_path": self.world_yaml,
            # "run_info": self.run_info,
        })
        return d

    def save_network(self, net, path):
        if 'encoder_ticks' not in self.app_params:
            world = self.get_sample_world(delete_rss=False)
            self.app_params.update({'encoder_ticks': world.population[0].neuro_tpc})
        super().save_network(net, path)

    def get_sample_world(self, delete_rss=True):
        cycles = self.cycles
        self.cycles = 0
        proc = caspian.Processor(self.processor_params)
        template_net = make_template(proc, self.n_inputs, self.n_outputs)
        world = self.simulate(proc, template_net)
        self.cycles = cycles
        if delete_rss:
            self.delete_rss()
        return world

    def delete_rss(self):
        if 'rss' in globals():
            del rss

    def save_artifacts(self, evolver, *args, **kwargs):
        if super().save_artifacts(evolver, *args, **kwargs) is None:
            return
        self.p.save_yaml_artifact("env.yaml", self.get_sample_world(delete_rss=False))
        self.delete_rss()

    def get_env_info(self):
        d = super().get_env_info()
        try:
            novel_swarms_path = envt.module_editable_path('novel_swarms')
            d['.dependencies'].update({
                'novel_swarms': {
                    'path': str(novel_swarms_path.resolve()),
                    "branch": envt.get_branch_name(novel_swarms_path),
                    "HEAD": envt.git_hash(novel_swarms_path),
                    "status": [s.strip() for s in envt.git_porcelain(novel_swarms_path).split('\n')],
                    'version': envt.get_module_version('novel_swarms'),
                },
            })
        except Exception:
            d['.dependencies'].update({'novel_swarms': envt.get_module_version('novel_swarms')})
        return d



def run(app, args):

    # Set up simulator and network

    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)
        net = app.net

    # Run app and print fitness
    world = app.simulate(proc, net)
    fitness = app.extract_fitness(world)
    print(f"Fitness: {fitness:8.4f}")

    if args.log_trajectories:
        graphing.plot_multiple(world)
        graphing.export(world, output_file=app.p.ensure_file_parents("agent_trajectories.xlsx"))
        # TODO: handle when no project

    return fitness


def test(app, args):

    # Set up simulator and network
    proc = caspian.Processor(app.processor_params)
    net = app.net

    if args.positions:
        from rss.rss2 import PredefinedInitialization, SCALE
        import pandas as pd
        fpath = args.positions

        with open(fpath, 'rb') as f:
            xlsx = f.read()
        xlsx = pd.ExcelFile(BytesIO(xlsx))
        sheets = xlsx.sheet_names

        n_runs = len(sheets)

        pinit = PredefinedInitialization()  # num_agents isn't used yet here

        def setup_i(i):
            pinit.set_states_from_xlsx(args.positions, sheet_number=i)
            pinit.rescale(SCALE)

            def setup(world):
                world.init_type = pinit
                return world

            return setup

        # Run app and print fitness
        fitnesses = [app.fitness(proc, net, setup_i(i)) for i in tqdm(range(n_runs))]

        # print(f"Fitness: {fitness:8.8f}")
        print(fitnesses)
        return fitnesses
    else:
        raise ArgumentError(args.positions, "Positions not specified")


def get_parsers(parser, subpar):
    # this is a separate function so we can inherit options from this module
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        sub.add_argument('--agent_yaml', default="rss/turbopi-milling/turbopi.yaml",
                         type=str, help="path to yaml config for agent")
        sub.add_argument('--world_yaml', default="rss/turbopi-milling/world.yaml",
                         type=str, help="path to yaml config for world")

    # for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
    #     pass  # sp[key].add_argument()

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['train'].add_argument('--logfile', default=None,
                             help="running log file path. By default, this is saved to the projectdir/training.log or tenn2_train.log for non-project mode.")  # noqa: E501

    sp['run'].add_argument('--track_history', action='store_true',
                           help="pass this to enable sensor vs. output plotting.")
    sp['run'].add_argument('--log_trajectories', action='store_true',
                           help="pass this to log sensor vs. output to file.")
    sp['run'].add_argument('--start_paused', action='store_true',
                           help="pass this to pause the simulation at startup. Press Space to unpause.")

    # Testing args
    sp['test'].add_argument('--positions', default=None,
                             help="file containing agent positions")
    sp['test'].add_argument('-p', '--processes', type=int, default=1,
                           help="number of threads for concurrent fitness evaluation.")

    return parser, subpar


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "connorsim_snn_eons-v01"  # type: ignore[reportAttributeAccessIssue]
    if args.project is None and args.logfile is None:
        args.logfile = "tenn2_train.log"

    app = ConnorMillingExperiment(args)

    # Do the appropriate action
    if args.action == "train":  # type: ignore[reportAttributeAccessIssue]
        common.experiment.train(app, args)
    elif args.action == "test":  # type: ignore[reportAttributeAccessIssue]
        test(app, args)
    elif args.action == "run":  # type: ignore[reportAttributeAccessIssue]
        run(app, args)
    else:
        raise RuntimeError("No action selected")


if __name__ == "__main__":
    main()

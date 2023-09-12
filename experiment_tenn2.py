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
from common.experiment import TennExperiment
import common.experiment

from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspian


class CustomPool():
    """pool class for Evolver, so we can use tqdm for those sweet progress bars"""

    def __init__(self, **tqdm_kwargs):  # max_workers=args.threads
        self.kwargs = tqdm_kwargs

    def map(self, fn, *iterables):
        return process_map(fn, *iterables, **self.kwargs)


class ConnorMillingExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    def __init__(self, args):
        super().__init__(args)
        self.agent_yaml = args.agent_yaml
        self.world_yaml = args.world_yaml
        self.run_info = None

        self.n_inputs, self.n_outputs, _, _ = MazeAgentCaspian.get_default_encoders(self.app_params['proc_ticks'])

        self.log("initialized experiment_tenn2")

    def fitness(self, processor, network):
        import rss2
        # setup sim

        network.set_data("processor", self.processor_params)

        robot_config = rss2.configure_robots(network, agent_yaml_path=self.agent_yaml, track_all=self.viz)
        world_config = rss2.configure_env(robot_config=robot_config, world_yaml_path=self.world_yaml,
                                          num_agents=self.agents, stop_at=self.sim_time)

        reward_history = []

        def get_how_many_on_goal(world):
            return sum([int(world.goals[0].agent_achieved_goal(agent)) for agent in world.population])

        def callback(world, screen):
            nonlocal reward_history
            # reward_history.append(get_how_many_on_goal(world))

            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.neuron_ids,
                    "Event Counts": a.neuron_counts
                })

        if self.viz is None:
            gui = rss2.TennlabGUI(x=world_config.w, y=0, h=world_config.h, w=200)
        elif self.viz is False:
            gui = False
        else:
            raise NotImplementedError

        world_subscriber = rss2.WorldSubscriber(func=callback)
        world_output = rss2.simulator(  # noqa run simulator
            world_config=world_config,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
        )

        # print(f"final count: {get_how_many_on_goal(world)}")
        # self.run_info = reward_history
        # return reward_history[-1]
        self.run_info = world_output.behavior[0].value_history
        return world_output.behavior[0].out_current()[1]


def main():
    parser, subpar = common.experiment.get_parsers()
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        sub.add_argument('--agent_yaml', default="../RobotSwarmSimulator/demo/configs/flockbots-icra-milling/flockbot.yaml",
                         type=str, help="path to yaml config for agent")
        sub.add_argument('--world_yaml', default="../RobotSwarmSimulator/demo/configs/flockbots-icra-milling/world.yaml",
                         type=str, help="path to yaml config for world")

    for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
        sp[key].add_argument('--network', help="network", default="networks/experiment_tenn2.json")

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['train'].add_argument('--network', default="networks/experiment_tenn2_train.json",
                             help="output network file path.")
    sp['train'].add_argument('--logfile', default="tenn2_train.log",
                             help="running log file path.")

    args = parser.parse_args()

    args.environment = "connorsim_snn_eons-v01"

    app = ConnorMillingExperiment(args)

    # Do the appropriate action
    if args.action == "train":
        common.experiment.train(app, args)
    else:
        if args.noviz:
            args.viz = False
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

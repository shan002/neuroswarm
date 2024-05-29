from typing import override
import math
# import matplotlib.pyplot as plt

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment

from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspianConfig
from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspian
from novel_swarms.behavior import AgentsAtGoal


class ConnorMillingExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    def __init__(self, args):
        super().__init__(args)
        self.agent_yaml = args.agent_yaml
        self.world_yaml = args.world_yaml
        self.run_info = None

        self.n_inputs, self.n_outputs, _, _ = MazeAgentCaspian.get_default_encoders(self.app_params['proc_ticks'])

        self.log("initialized experiment_tenn5")

    @override
    def fitness(self, processor, network):
        import rss2
        # setup sim

        network.set_data("processor", self.processor_params)

        robot_config = rss2.configure_robots(network, MazeAgentCaspianConfig, agent_yaml_path=self.agent_yaml, track_all=self.viz)
        world = rss2.create_environment(robot_config=robot_config, world_yaml_path=self.world_yaml,
                                          num_agents=self.agents, stop_at=self.sim_time)
        world.behavior = [AgentsAtGoal()]

        reward_history = []

        def get_how_many_on_goal(world):
            return sum([int(world.goals[0].agent_achieved_goal(agent)) for agent in world.population])

        def callback(world, screen):
            nonlocal reward_history
            reward_history.append(get_how_many_on_goal(world))

            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.neuron_ids,
                    "Event Counts": a.neuron_counts
                })

        gui = rss2.TennlabGUI(x=world.w, y=0, h=world.h, w=200)
        if self.viz is False or self.noviz:
            gui = False

        world_subscriber = rss2.WorldSubscriber(func=callback)
        world_output = rss2.simulator(  # type:ignore[reportPrivateLocalImportUsage]  # run simulator
            world_config=world,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
        )

        # print(f"final count: {get_how_many_on_goal(world)}")
        self.run_info = reward_history
        n = len(reward_history)
        t_firstgoal, _ = next((i, r) for i, r in enumerate(reward_history) if r > 0)
        earlybird_reward = t_firstgoal / n * 10000
        # weights = [x / n for x in range(n, 0, -1)]
        # weighted_rewards = [r * w for r, w in zip(reward_history, weights)]
        return earlybird_reward + sum(reward_history) / 1000 + reward_history[-1] * 1000


def get_parsers(parser, subpar):
    # this is a separate function so we can inherit options from this module
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        sub.add_argument('--agent_yaml', default="../RobotSwarmSimulator/demo/configs/flockbots-icra/goalbot.yaml",
                         type=str, help="path to yaml config for agent")
        sub.add_argument('--world_yaml', default="../RobotSwarmSimulator/demo/configs/flockbots-icra/world.yaml",
                         type=str, help="path to yaml config for world")

    for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
        sp[key].add_argument('--network', help="network", default="networks/experiment_tenn5.json")

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['train'].add_argument('--network', default="networks/experiment_tenn5_train.json",
                             help="output network file path.")
    sp['train'].add_argument('--logfile', default="tenn5_train.log",
                             help="running log file path.")

    return parser, subpar


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "connorsim_snn_eons_goalsearch-v01"  # type:ignore[reportAttributeAccessIssue]

    app = ConnorMillingExperiment(args)

    # Do the appropriate action
    if args.action == "train":  # type:ignore[reportAttributeAccessIssue]
        common.experiment.train(app, args)
    else:
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

# from multiprocessing import Pool, TimeoutError
# from tqdm.contrib.concurrent import process_map
# import neuro
# import caspian
# import random
# import os
# import time
# import pathlib
# import matplotlib.pyplot as plt

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment

from zss.flockbot_caspian import FlockbotCaspian


class ZespolExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Shay Zespol


    """

    def __init__(self, args):
        super().__init__(args)
        self.n_inputs, self.n_outputs, _, _ = FlockbotCaspian.get_default_encoders(self.app_params['proc_ticks'])
        self.log("initialized experiment_tenn3")

    def fitness(self, processor, network):
        metrics = self.run(processor, network)
        return metrics["Circliness"]

    def run(self, processor, network):
        import zss
        # setup sim

        network.set_data("processor", self.processor_params)

        zespol_config = {
            "num_agents": self.agents,
            "ticks": self.cycles,
            "ticks_per_second": 7.5,
            "viz": self.viz,
            "network": network,
            "neuro_tpc": self.app_params['proc_ticks'],
            "world_seed": 2023,
        }

        world = zss.setup_world(zespol_config)

        for t in range(self.cycles):
            world.tick()
            metrics = world.metric_window[-1]

        if self.viz:
            world.visualizer.compile_videos()

        # print(f"final count: {get_how_many_on_goal(world)}")
        # self.run_info = reward_history
        # return reward_history[-1]
        return metrics


def main():
    parser, subpar = common.experiment.get_parsers()
    sp = subpar.parsers

    # for sub in sp.values():  # applies to everything
    #     sub.add_argument('--agent_yaml', default="../RobotSwarmSimulator/demo/configs/flockbots-icra-milling/flockbot.yaml",
    #                      type=str, help="path to yaml config for agent")

    for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
        sp[key].add_argument('--network', help="network", default="networks/experiment_tenn3.json")

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['train'].add_argument('--network', default="networks/experiment_tenn3_train.json",
                             help="output network file path.")
    sp['train'].add_argument('--logfile', default="tenn3_train.log",
                             help="running log file path.")

    args = parser.parse_args()

    args.environment = "zespol_snn_eons-v01"

    app = ZespolExperiment(args)

    # Do the appropriate action
    if args.action == "train":
        common.experiment.train(app, args)
    else:
        if args.noviz:
            args.viz = False
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

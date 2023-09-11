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
from common.experiment import TennExperiment, get_parser, train, run

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


def main(args):
    args.environment = "zespol_snn_eons-v01"

    app = ZespolExperiment(args)

    # Do the appropriate action
    if args.action == "train":
        train(app, args)
    else:
        if args.noviz:
            args.viz = False
        run(app, args)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

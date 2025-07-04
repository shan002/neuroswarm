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

from distributed import Client

from zss.flockbot_caspian import FlockbotCaspian

from leap_ec.simple import ea_solve


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



        # print(f"final count: {get_how_many_on_goal(world)}")
        # self.run_info = reward_history
        # return reward_history[-1]
        return metrics
    
def evaluate_zespol(controller) -> float:
    import zss

    num_agents = 10
    sim_time = 1000
    viz: bool = False

    zespol_config = {
        "num_agents": num_agents,
        "ticks": sim_time,
        "ticks_per_second": 7.5,
        "viz": viz,
        "controller": controller,
        "world_seed": 2023,
    }

    print(0)

    # Kevin - it seems to die here for some reason.
    world = zss.setup_world(zespol_config)

    print(1)

    for t in range(sim_time):
        world.tick()
        metrics = world.metric_window[-1]

    if viz:
        world.visualizer.compile_videos()

    return metrics


def main():
    parser, subpar = common.experiment.get_parsers()
    sp = subpar.parsers

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['train'].add_argument('--logfile', default="tenn4-test-train.log",
                             help="running log file path.")

    args = parser.parse_args()

    args.environment = "zespol_snn_eons-v01"

    # Kevin: You can add argparse arguments here if you want
    num_generations: int = 100
    population_size: int = 25
    bounds:list[tuple] =[(-1, 1) for _ in range(4)]
    mutation_std: float = 0.1
    final_population = ea_solve(
        evaluate_zespol,
        generations=num_generations,
        pop_size=population_size,
        bounds=bounds,
        maximize=True,
        viz=False,
        mutation_std=mutation_std,
        dask_client=Client()
    )

    print(final_population)


if __name__ == "__main__":
    main()

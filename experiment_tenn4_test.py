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

import numpy as np

from zss.flockbot_caspian import FlockbotCaspian

from leap_ec.simple import ea_solve

    
def evaluate_zespol(controller) -> float:
    #print(controller)

    return np.random.random()


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

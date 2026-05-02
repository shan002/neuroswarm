import copy
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm.contrib.concurrent import process_map

from swarmsim.config import register_dictlike_type, register_agent_type
from swarmsim.agent.MazeAgent import MazeAgentConfig
from swarmsim.world.spawners.DonutSpawner import DonutAgentSpawner
from swarmsim.world.RectangularWorld import RectangularWorldConfig
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
from swarmsim.world.simulate import main as simulator
# from ..gui import TennlabGUI

register_dictlike_type('spawners', 'DonutAgentSpawner', DonutAgentSpawner)

cwd = Path(__file__).resolve().parent
config = RectangularWorldConfig.from_yaml(cwd / "world.yaml")

# gui = TennlabGUI(x=0, y=0, h=0, w=300)
# gui.position = "sidebar_right"


def stop(world):
    return any(m.current_value for m in world.metrics)


def test_single(config):
    tempconfig = copy.deepcopy(config)
    # config.seed += i
    stats = Counter()
    world = simulator(
        world_config=tempconfig,
        subscribers=[],
        # gui=gui,
        show_gui=False,
        start_paused=False,
        stop_detection=stop,
    )  # run simulator
    for m in world.metrics:
        stats[m.name] += m.current_value
    return stats


def test_mp(n=100):
    configs = []
    seeds = np.random.default_rng(getattr(config, "seed", None)).integers(0, 2**31, size=n)
    for i in range(100):
        tempconfig = copy.deepcopy(config)
        tempconfig.seed = seeds[i]
        configs.append(tempconfig)
    results = process_map(test_single, configs)
    print(sum(results, Counter()))


def run():
    world = simulator(
        world_config=config,
        subscribers=[],
        # gui=gui,
        show_gui=True,
        start_paused=True,
        # stop_detection=stop,
    )  # run simulator
    for m in world.metrics:
        print(f"{m.name}: {m.current_value}")


if __name__ == "__main__":
    test_mp()
    # run()

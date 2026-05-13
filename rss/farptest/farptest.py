import copy
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm.contrib.concurrent import process_map
import seaborn as sns
from matplotlib import pyplot as plt

from swarmsim.config import register_dictlike_type, register_agent_type
from swarmsim.agent.MazeAgent import MazeAgentConfig
from swarmsim.world.spawners.DonutSpawner import DonutAgentSpawner
from swarmsim.world.RectangularWorld import RectangularWorldConfig
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
from swarmsim.world.simulate import main as simulator


cwd = Path(__file__).resolve().parent
config = RectangularWorldConfig.from_yaml(cwd / "world.yaml")

# gui = TennlabGUI(x=0, y=0, h=0, w=300)
# gui.position = "sidebar_right"


def stop(world):
    return any(m.current_value for m in world.metrics) or world.total_steps > 3000


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
        stats[m.name] += m.value
        if m.name == "Time to capture":
            ttc = m.value
    return stats, ttc


def test_mp(samples=100):
    seeds = np.random.default_rng(getattr(config, "seed", None)).integers(0, 2**31, size=samples)
    results = []
    for n in range(1, 10):
        configs = []
        for i in range(samples):
            tempconfig = copy.deepcopy(config)
            tempconfig.seed = seeds[i]
            tempconfig.spawners[0]['n'] = n
            configs.append(tempconfig)
        ret_arr = process_map(test_single, configs)
        stats, ttcs = zip(*ret_arr)
        print('n: ', n, sum(stats, Counter()))
        for run in stats:
            run['Time to capture'] = -1000 if run['Time to capture'] == 0 else run['Time to capture']
            results.append({
                'n': n,
                **run
            })
    # ttcs = np.array(ttcs)
    # ttcs[ttcs == 0] = -1000
    # sns.histplot(ttcs)
    # plt.show()
    print(results)
    return results


def run():
    world = simulator(
        world_config=config,
        subscribers=[],
        # gui=gui,
        show_gui=True,
        start_paused=True,
        stop_detection=stop,
    )  # run simulator
    for m in world.metrics:
        print(f"{m.name}: {m.current_value}")


if __name__ == "__main__":
    # test_mp()
    run()

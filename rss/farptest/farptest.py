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
config = RectangularWorldConfig.from_yaml(cwd / "ttc.yaml")

# gui = TennlabGUI(x=0, y=0, h=0, w=300)
# gui.position = "sidebar_right"


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
    )  # run simulator
    for m in world.metrics:
        stats[m.name] += m.value
    out = world.metrics[0].value
    return stats, out


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
        for stat in stats:
            results.append({
                'n': n,
                **stat
            })
    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)
    return results


def test_grid(samples=100):
    seeds = np.random.default_rng(getattr(config, "seed", None)).integers(0, 2**31, size=samples)
    results = []
    x, y = np.meshgrid(
        np.linspace(0.0, 0.3, 12),
        np.linspace(0.0, 0.6, 7),
    )
    configs = []
    n = 6
    for v, w in zip(x.flatten(), y.flatten()):
        for i in range(samples):
            tempconfig = copy.deepcopy(config)
            tempconfig.seed = seeds[i]
            tempconfig.spawners[0]['n'] = n
            controller = tempconfig.spawners[0]['agent']['controller']
            controller['a'] = [v, w]
            controller['b'] = [v, -w]
            configs.append(tempconfig)
    ret_arr = process_map(test_single, configs)
    stats, ttcs = zip(*ret_arr)
    print('n: ', n, sum(stats, Counter()))
    for stat, cfg in zip(stats, configs):
        results.append({
            'n': n,
            'v': cfg.spawners[0]['agent']['controller']['a'][0],
            'w': cfg.spawners[0]['agent']['controller']['a'][1],
            **stat
        })
    # ttcs = np.array(ttcs)
    # sns.histplot(ttcs)
    # plt.show()
    print(results)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('grid.csv')
    return results


def run():
    world = simulator(
        world_config=config,
        subscribers=[],
        # gui=gui,
        show_gui=True,
        start_paused=True,
    )  # run simulator
    for m in world.metrics:
        print(f"{m.name}: {m.current_value}")
    return world


if __name__ == "__main__":
    # test_mp()
    # test_grid()
    run()
    # print(*test_single(config))

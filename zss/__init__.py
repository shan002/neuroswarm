import argparse
from math import pi as PI
from math import degrees
import numpy as np
import random

from swarms.lib.core.objects.agent import Agent
from swarms.lib.core.objects.swarm import Swarm
from swarms.lib.core.objects.world import World
from swarms.lib.core.states.swarm_state import SwarmState
# from swarms.lib.utils import generate_spawn_points
from swarms.lib.sensors.binary import BinarySensor

from .metrics import Circliness

from .flockbot_caspian import FlockbotCaspian
# from .flockbot_binarycontroller import FlockbotBinarycontroller

# typing
from typing import List
from .metrics import Metric
from swarms.lib.core.states.agent_state import AgentState


class CustomSwarmState(SwarmState):
    def __init__(self, agent_states: list, num_processes: int, world_radius: float, handlers):
        self.agent_states: list[AgentState] = agent_states
        self.num_processes: int = num_processes
        self.world_radius: float = world_radius

        self.metric_handlers = handlers

        self.metrics = {}
        for metric in self.metric_handlers:
            metric.population = agent_states  # update metric handler with new info
            metric.calculate()
            name, value = metric.out_current()
            if name in self.metrics:
                raise KeyError(f"Refusing to add metric: Metrics dict already contains metric with name {name}")
            self.metrics.update({name: value})


class CustomSwarm(Swarm):
    def __init__(self, agents, spt: float, num_processes=1, metric_handlers: List[Metric] = None):
        super().__init__(agents, spt, num_processes)
        self.metric_handlers = metric_handlers

    def get_state(self, world_radius) -> CustomSwarmState:
        states = [agent.get_state() for agent in self.agents]
        return CustomSwarmState(
            agent_states=states,
            num_processes=self.num_processes,
            world_radius=world_radius,
            handlers=self.metric_handlers,
        )


def generate_initial_SE2(rng, x1, y1, x2, y2):
    while True:
        yield rng.uniform(x1, x2), rng.uniform(y1, y2), rng.uniform(0, 2) * PI


def generate_initial_SE2_packed(rng, x1, y1, x2, y2, r):
    agents = []

    gen = generate_initial_SE2(rng, x1, y1, x2, y2)

    def dist(a, b):
        d = np.array(a[:2]) - np.array(b[:2])
        return float(np.linalg.norm(d))

    while True:
        agents.append(gen.__next__())
        for i in range(999 + len(agents) ** 2):
            agent = agents[-1]
            overlap = [dist(agent, other) < 2 * r for other in agents if other is not agent]
            if any(overlap):
                agents[-1] = gen.__next__()
            else:
                yield agent
                break
        else:
            raise StopIteration


def setup_world(config: dict) -> None:
    """
    serve as the epicenter of the entire experiment

    Arguments:
    ----------
    1) config (dict): various parameters that modify the program's
        runtime attributes

    Returns:
    --------
    N/A
    """
    default_config = {
        "name": "Flockbot Simulation",
        'num_agents': 10,
        'ticks': 2000,
        'ticks_per_second': 7.5,
        'viz': None,
    }
    default_config.update(config)
    config = default_config

    name = config['name']
    num_agents: int = config['num_agents']
    ticks_max: int = config['ticks']
    spt: float = 1 / config['ticks_per_second']

    world_size = np.array([15, 15])
    network = config['network']
    viz = config["viz"]

    viz_config: dict = {
        "experiment_name": name,
        "fps": 7.5,
        "log_dir": "./logs/",
        "enabled": bool(viz)
    }
    viz_config['experiment_name'] = f"flockbots_milling_{num_agents}n_caspian"

    sensor_config = {
        "view_dist": 3.6,  # meters
        "fov": 0.4,
    }

    world_seed = config['world_seed']
    if world_seed is None:
        rng = random.Random()
    elif isinstance(world_seed, [int, float]):
        rng = random.Random(world_seed)
    else:
        rng = world_seed
    gen = generate_initial_SE2_packed(rng, 1.9, 1.9, 3.1, 3.1, FlockbotCaspian.agent_radius)
    initial_states = [gen.__next__() for i in range(num_agents)]

    agents: list[Agent] = []
    for i, state in enumerate(initial_states):
        x, y, theta = state
        sensor = BinarySensor(**sensor_config)
        agent = FlockbotCaspian(
            name=f'agent{i}',
            pos=(x, y),
            heading=theta,
            spt=spt,
            sensors=[sensor],
            network=network,
            neuro_tpc=config['neuro_tpc']
        )
        # agent = FlockbotBinarycontroller(
        #     name=f'agent{i}',
        #     pos=(x, y),
        #     heading=theta,
        #     spt=spt,
        #     sensors=[sensor],
        # )
        agents.append(agent)

    # setup metrics
    circliness = Circliness(history_size=ticks_max, avg_history_max=450)

    swarm = CustomSwarm(
        agents=agents,
        spt=spt,
        metric_handlers=[circliness],
    )

    world = World(
        size=world_size,
        spt=spt,
        swarm=swarm,
        viz_config=viz_config
    )

    return world


if __name__ == "__main__":
    config = {
        'num_agents': 10,
        'ticks': 2000,
        'ticks_per_second': 7.5,
    }
    # for i in range(7):
    #     for j in range(8):
    #         config['max_velocity'] = 0.025 + 0.025 * i
    #         config['turning_rate'] = 0.25 + 0.25 * j
    world = setup_world(config)

    print(world)

    print("nothing executed as main")

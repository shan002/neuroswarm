import argparse
import math
from math import pi as PI
import numpy as np
import random

from ..objects.epuck_agent import EPuckRobot
from swarms.lib.core.objects.agent import Agent
from swarms.lib.core.objects.swarm import Swarm
from swarms.lib.core.objects.world import World
from swarms.lib.utils import generate_spawn_points

from flockbotcaspian import FlockbotCaspian


def parse_args() -> dict:
    """
    parse various command line arguments from the user

    Arguments:
    ----------
    N/A

    Returns:
    --------
    1) config (dict): various parameters that modify the program's
        runtime attributes
    """
    parser = argparse.ArgumentParser(
        description="Base E-Puck Robot Test"
    )
    parser.add_argument(
        '-n',
        '--num_agents',
        default=10,
        help="the number of agents in the swarm",
        type=int
    )
    parser.add_argument(
        '-t',
        '--ticks',
        default=2000,
        help="the number of ticks to run.",
        type=int
    )
    parser.add_argument(
        '--ticks_per_second',
        default=7.5,
        help="the number of simulation ticks equivalent to one second",
        type=float
    )

    config: argparse.Namespace = parser.parse_args()
    config: dict = vars(config)
    return config


def generate_initial_position(rng):
    yield rng.uniform(-1., 1.), rng.uniform(-1., 1.), rng.uniform(0, 1) * PI


def main(config: dict) -> None:
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
    sname: str = "Base Puck Test"
    print(f"--> {sname}: initializing configuration")
    num_agents: int = config['num_agents']
    ticks_max: int = config['ticks']
    sensor_distance: float = config['sensor_distance']
    sensor_fov: float = math.radians(config['sensor_fov'])
    spt: float = 1 / config['ticks_per_second']
    world_size = np.array([15, 15])
    viz_config: dict = {
        "experiment_name": "BaseEpuck",
        "fps": 7.5,
        "log_dir": "./logs/",
        "enabled": True
    }
    viz_config['experiment_name'] = f"flockbots_{num_agents}n_caspian"
    print(f"--> {sname}: initialized configuration")

    print(f"--> {sname}: generating spawn points")
    spawn_points: list = generate_spawn_points(
        space=world_size,
        num_points=num_agents
    )
    print(f"--> {sname}: generated {len(spawn_points)} spawn points")

    print(f"--> {sname}: initializing agents")
    agents: list[Agent] = []
    for i in range(num_agents):
        agent = EPuckRobot(
            x=(world_size[0] / 2) + (random.random() * 5),  # spawn_points[i][0],
            y=(world_size[1] / 2) + (random.random() * 5),  # spawn_points[i][1],
            yaw=math.pi / 8,  # (random.random() * 2 * math.pi),
            name=f'agent{i}',
            spt=spt,
            sensor_distance=sensor_distance,
            sensor_fov=sensor_fov,
            controller_params=controller_params,
            max_velocity=max_velocity
        )
        agents.append(agent)

    swarm = Swarm(
        agents=agents,
        spt=spt
    )

    world = World(
        size=world_size,
        spt=spt,
        swarm=swarm,
        viz_config=viz_config
    )

    if ticks_max < 0:
        while True:
            world.tick()
    else:
        for _ in range(ticks_max):
            world.tick()

    # world.visualizer.compile_videos()

    print(f"--> {sname}: closing simulation")


if __name__ == "__main__":
    config: dict = parse_args()
    for i in range(7):
        for j in range(8):
            config['max_velocity'] = 0.025 + 0.025 * i
            config['turning_rate'] = 0.25 + 0.25 * j
            main(config)


if __name__ == '__main__':
    print("nothing executed as main")

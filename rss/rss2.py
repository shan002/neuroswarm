"""
Example Script
The SwarmSimulator allows control of the world and agents at every step within the main loop
"""
# import random

# Import Agent embodiments
# from swarmsim.config.AgentConfig import *
# from swarmsim.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
# from swarmsim.agent.MazeAgentCaspian import MazeAgentCaspianConfig
# from swarmsim.agent.MillingAgentCaspian import MillingAgentCaspianConfig

# Import FOV binary sensor
from swarmsim.sensors.SensorFactory import SensorFactory

# Import Rectangular World Data, Starting Region, and Goal Region
from swarmsim.config.WorldConfig import WorldYAMLFactory
# from swarmsim.config.WorldConfig import RectangularWorldConfig
# from swarmsim.world.goals.Goal import CylinderGoal
# from swarmsim.world.initialization.RandomInit import RectRandomInitialization
from swarmsim.world.initialization.PredefInit import PredefinedInitialization as PredefinedInitialization

# Import a world subscriber, that can read/write to the world data at runtime
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber

# Import the Behavior Measurements (Metrics) that can measure the agents over time
from swarmsim.metrics import RadialVariance
from swarmsim.metrics import AgentsAtGoal
# from swarmsim.metrics import Circliness

# Import the custom Controller Class
from swarmsim.agent.control.Controller import Controller

# Import the simulation loop
from swarmsim.world.simulate import main as simulator

simulator = simulator  # explicit export

SCALE = 1  # Set the conversion factor for Body Lengths to pixels (all metrics will be scaled appropriately by this value)


def configure_robots(
    network,
    agent_config_class,
    agent_yaml_path,
    seed=None,
    track_all=None,
    track_io=False,
    scale=SCALE,
):
    """
    Select the Robot's Sensors and Embodiment, return the robot configuration
    """
    # example agent_yaml file: "RobotSwarmSimulator/demo/configs/flockbots-icra-milling/flockbot.yaml"

    # Import the config from YAML
    import yaml
    config = None
    with open(agent_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config["sensors"] = SensorFactory.create(config["sensors"])

    config["network"] = network
    config["neuro_track_all"] = bool(track_all)
    config["track_io"] = track_io

    agent_config = agent_config_class(**config)

    # if seed is not None:
    #     normal_flockbot.seed = seed

    # Uncomment to remove FN/FP from agents (Testing)
    # normal_flockbot.sensors.sensors[0].fn = 0.0
    # normal_flockbot.sensors.sensors[0].fp = 0.0

    agent_config.rescale(scale)  # Convert all the BodyLength measurements to pixels in config
    return agent_config


def establish_goal_metrics():
    # metric_0 = PercentageAtGoal(0.01)  # Record the time (timesteps) at which 1% of the agents found the goal
    # metric_A = PercentageAtGoal(0.5)  # Record the time (timesteps) at which 50% of the agents found the goal
    # metric_B = PercentageAtGoal(0.75)  # Record the time (timesteps) at which 75% of the agents found the goal
    # metric_C = PercentageAtGoal(0.90)  # Record the time (timesteps) at which 90% of the agents found the goal
    # metric_D = PercentageAtGoal(1.00)  # Record the time (timesteps) at which 100% of the agents found the goal
    metric_E = AgentsAtGoal()  # Record the number of Agents in the Goal Region
    # return [metric_0, metric_A, metric_B, metric_C, metric_D, metric_E]
    return [metric_E]


def establish_milling_metrics():
    # TODO: Update this value with Kevin's Formulation
    circliness = RadialVariance()
    return [circliness]


def create_environment(robot_config, world_yaml_path, num_agents=20, seed=None, stop_at=9999, scale=SCALE):
    # search_and_rendezvous_world = WorldYAMLFactory.from_yaml("demo/configs/flockbots-icra/world.yaml")

    # Import the world data from YAML
    world_cfg = RectangularWorldConfig.from_yaml(world_yaml_path)
    world_cfg.addAgentConfig(robot_config)
    world_cfg.population_size = num_agents
    world_cfg.stop_at = stop_at
    world_cfg.factor_zoom(scale)
    if seed is not None:
        world_cfg.seed = seed
    return world_cfg

    # Create a Goal for the Agents to find
    # goal_region = CylinderGoal(
    #     x=400,  # Center X, in pixels
    #     y=100,  # Center y, in pixels
    #     r=8.5,  # Radius of physical cylinder object
    #     range=40.0  # Range, in pixels, at which agents are "counted" as being at the goal
    # )


# it is recommended to use scale=1
def generate_positions(world_yaml_path, num_agents=20, seed=None, scale=SCALE):
    world_cfg = WorldYAMLFactory.from_yaml(world_yaml_path)
    world_cfg.population_size = num_agents
    if seed is not None:
        world_cfg.init_type.seed = seed  # override seed
    world_cfg.factor_zoom(scale)
    return world_cfg.init_type.positions


def generate_position_sets(world_yaml_path, num_agents=20, seeds=None, scale=SCALE):
    if seeds is None:
        seeds = []
    return [
        (seed,
        generate_positions(world_yaml_path=world_yaml_path, num_agents=num_agents, seed=seed, scale=scale))
        for seed in seeds
    ]


def save_position_sets_to_xlsx(position_sets, output_file: str):
    import pandas as pd
    with pd.ExcelWriter(output_file) as writer:
        for name, positions in position_sets:
            df = pd.DataFrame(positions, columns=['x', 'y', 'angle (rads from east)'])
            df.to_excel(writer, sheet_name=f'{name}')

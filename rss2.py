"""
Example Script
The SwarmSimulator allows control of the world and agents at every step within the main loop
"""
# import random

# Import Agent embodiments
# from novel_swarms.config.AgentConfig import *
# from novel_swarms.config.HeterogenSwarmConfig import HeterogeneousSwarmConfig
from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspianConfig
from novel_swarms.agent.MillingAgentCaspian import MillingAgentCaspianConfig

# Import FOV binary sensor
from novel_swarms.sensors.SensorFactory import SensorFactory

# Import Rectangular World Data, Starting Region, and Goal Region
from novel_swarms.config.WorldConfig import WorldYAMLFactory
# from novel_swarms.config.WorldConfig import RectangularWorldConfig
# from novel_swarms.world.goals.Goal import CylinderGoal
# from novel_swarms.world.initialization.RandomInit import RectRandomInitialization

# Import a world subscriber, that can read/write to the world data at runtime
from novel_swarms.world.subscribers.WorldSubscriber import WorldSubscriber

# Import the Behavior Measurements (Metrics) that can measure the agents over time
from novel_swarms.behavior import RadialVarianceBehavior, AgentsAtGoal, Circliness

# Import the custom Controller Class
from novel_swarms.agent.control.Controller import Controller

# Import the simulation loop
from novel_swarms.world.simulate import main as simulator

from novel_swarms.gui.agentGUI import DifferentialDriveGUI, pygame, np

SCALE = 10  # Set the conversion factor for Body Lengths to pixels (all metrics will be scaled appropriately by this value)


class TennlabGUI(DifferentialDriveGUI):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    # def set_selected(self, agent: MazeAgentCaspian):
    #     super().set_selected(agent)

    def draw(self, screen):
        # super().draw(screen)
        self.text_baseline = 10
        if pygame.font:
            if self.title:
                self.appendTextToGUI(screen, self.title, size=20)
            if self.subtitle:
                self.appendTextToGUI(screen, self.subtitle, size=18)

            self.appendTextToGUI(screen, f"Timesteps: {self.time}")
            if self.selected:
                a = self.selected
                self.appendTextToGUI(screen, f"Current Agent: {a.name}")
                self.appendTextToGUI(screen, f"")
                self.appendTextToGUI(screen, f"x: {a.get_x_pos()}")
                self.appendTextToGUI(screen, f"y: {a.get_y_pos()}")
                self.appendTextToGUI(screen, f"dx: {a.dx}")
                self.appendTextToGUI(screen, f"dy: {a.dy}")
                self.appendTextToGUI(screen, f"sense-state: {a.get_sensors().getState()}")
                if hasattr(a, "i_1") and hasattr(a, "i_2"):
                    self.appendTextToGUI(screen, f"Idio_1: {a.i_1}")
                    self.appendTextToGUI(screen, f"Idio_2: {a.i_2}")
                self.appendTextToGUI(screen, f"")
                if hasattr(a, "controller"):
                    self.appendTextToGUI(screen, f"controller: {a.controller}")
                    self.appendTextToGUI(screen, f"")
                self.appendTextToGUI(screen, f"θ: {a.angle % (2 * np.pi)}")
                if hasattr(a, "agent_in_sight") and a.agent_in_sight is not None:
                    self.appendTextToGUI(screen, f"sees: {a.agent_in_sight.name}")
                try:
                    v, w = a.requested
                except AttributeError:
                    pass
                else:
                    self.appendTextToGUI(screen, f"ego v (bodylen): {v}")
                    self.appendTextToGUI(screen, f"ego v   (m/s): {v * 0.151}")
                    self.appendTextToGUI(screen, f"ego ω (rad/s): {w}")
                if a.neuron_counts is not None:
                    self.appendTextToGUI(screen, f"outs: {a.neuron_counts}")
            else:
                self.appendTextToGUI(screen, "Current Agent: None")
                self.appendTextToGUI(screen, "")
                self.appendTextToGUI(screen, "Behavior", size=18)
                for b in self.world.behavior:
                    out = b.out_current()
                    b.draw(screen)
                    try:
                        self.appendTextToGUI(screen, "{} : {:0.3f}".format(out[0], out[1]))
                    except ValueError:
                        pass
                    except Exception:
                        self.appendTextToGUI(screen, "{} : {}".format(out[0], out[1]))
        else:
            print("NO FONT")


def configure_robots(network, agent_yaml_path, seed=None, track_all=None):
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

    # agent_config = MazeAgentCaspianConfig(**config)
    agent_config = MillingAgentCaspianConfig(**config)
    agent_config.controller = Controller('self')

    if seed is not None:
        normal_flockbot.seed = seed

    # Uncomment to remove FN/FP from agents (Testing)
    # normal_flockbot.sensors.sensors[0].fn = 0.0
    # normal_flockbot.sensors.sensors[0].fp = 0.0

    agent_config.rescale(SCALE)  # Convert all the BodyLength measurements to pixels in config
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
    circliness = RadialVarianceBehavior()
    return [circliness]


def configure_env(robot_config, world_yaml_path, num_agents=20, seed=None, stop_at=9999):
    # search_and_rendezvous_world = WorldYAMLFactory.from_yaml("demo/configs/flockbots-icra/world.yaml")

    # Import the world data from YAML
    world = WorldYAMLFactory.from_yaml(world_yaml_path)
    world.addAgentConfig(robot_config)
    world.population_size = num_agents
    '''
        # Create a Goal for the Agents to find
        goal_region = CylinderGoal(
            x=400,  # Center X, in pixels
            y=100,  # Center y, in pixels
            r=8.5,  # Radius of physical cylinder object
            range=40.0  # Range, in pixels, at which agents are "counted" as being at the goal
        )
    '''
    world.factor_zoom(SCALE)
    if seed is not None:
        world.seed = seed
    world.stop_at = stop_at
    world.behavior = [Circliness(history=stop_at, avg_history_max=450)]
    return world

"""
Example Script
The SwarmSimulator allows control of the world and agents at every step within the main loop
"""
import random

# Import Agent embodiments
# from novel_swarms.config.AgentConfig import *
from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspianConfig

# Import FOV binary sensor
from novel_swarms.sensors.SensorSet import SensorSet
from novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor

# Import Rectangular World Data, Starting Region, and Goal Region
from novel_swarms.config.WorldConfig import RectangularWorldConfig
from novel_swarms.world.goals.Goal import CylinderGoal
from novel_swarms.world.initialization.RandomInit import RectRandomInitialization

# Import a world subscriber, that can read/write to the world data at runtime
from novel_swarms.world.subscribers.WorldSubscriber import WorldSubscriber

# Import the simulation loop
from novel_swarms.world.simulate import main as simulator

from novel_swarms.gui.agentGUI import DifferentialDriveGUI


class TennlabGUI(DifferentialDriveGUI):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    # def set_selected(self, agent: MazeAgentCaspian):
    #     super().set_selected(agent)

    def draw(self, screen):
        super().draw(screen)
        if self.selected:
            a = self.selected
            try:
                v, w = a.requested
            except AttributeError:
                pass
            else:
                self.appendTextToGUI(screen, f"ego v (bodylen): {v}")
                self.appendTextToGUI(screen, f"ego v   (m/s): {v * 0.151}")
                self.appendTextToGUI(screen, f"ego ω (rad/s): {w}")


def configure_robots(network, track_all=None):
    """
    Select the Robot's Sensors and Embodiment, return the robot configuration
    """
    sensors = SensorSet(
        [
            BinaryFOVSensor(
                theta=18,  # Angle of Vision / 2
                degrees=True,  # Indicate that the fov is defined in degrees, not radians
                distance=80,  # Detection Distance, in pixels
                show=True,  # Whether to show the sensor in the simulator
                detect_goal_with_added_state=True,
            ),
        ],
        custom_state_decision="Linear",
    )

    # import inspect
    # signature = inspect.signature(MazeAgentCaspianConfig.__init__).parameters
    # for name, parameter in signature.items():
    #     print(name, parameter.default, parameter.annotation, parameter.kind)
    goal_seeking_robot = MazeAgentCaspianConfig(
        sensors=sensors,  # Attach the previously defined sensors to the agent
        # Here, the controller is of the form [v_0, omega_0, v_1, omega_0, ...]
        network=network,
        agent_radius=4,  # Body radius, in pixels
        stop_at_goal=False,  # Don't automatically stop this robot when within goal region
        dt=0.13,  # Timestep value
        body_filled=True,  # Color in the body
        neuro_track_all=bool(track_all),
    )

    return goal_seeking_robot


def configure_env(robot_config, size=(500, 500), num_agents=20, stop_at=9999):
    """
    Select the World for the robots to interact in. Define the start region and goal region.

    Params:
    - robot_config: A robot configuration model (see configure_robots func)
    - size (optional, default: (500, 500)): A two-element tuple containing the WIDTH, HEIGHT of the world, in pixels
    - num_agents (optional, default: 20): The number of agents to instantiate in the environment

    Return: The World Configuration Data
    """

    SEED = None  # You may configure a world seed

    # Randomly Assign Agents to a x, y, $/theta$ orientation within the specified bounding box
    starting_region = RectRandomInitialization(
        num_agents=num_agents,
        bb=((50, 350), (150, 450))  # Spawn Bounding Box
    )

    # Create a Goal for the Agents to find
    goal_region = CylinderGoal(
        x=400,  # Center X, in pixels
        y=100,  # Center y, in pixels
        r=8.5,  # Radius of physical cylinder object
        range=40.0  # Range, in pixels, at which agents are "counted" as being at the goal
    )

    # Create the World for the Agents to interact in
    env = RectangularWorldConfig(
        size=size,
        agentConfig=robot_config,
        n_agents=num_agents,
        seed=SEED,
        init_type=starting_region,  # A starting region where agents will spawn at t=0
        goals=[goal_region],  # A list of goals for the robots to find
        stop_at=stop_at,
    )

    return env

# Create a Callback function that will be called at every .step() of the world
# If interfacing is complex enough that is cannot be done in a callback func, use


def callback(world, screen):
    """
    Read/Write from the world data

    Params:
    - world: A World object (see src/novel_swarms/world/) that contains agent information. Can be modified in-place.
    - screen: A pygame screen object that allows for direct read/write of the pixel values in the window
    """

    # Example: READ all agents (x, y, theta) positions.
    # positions = [(agent.x_pos, agent.y_pos, agent.angle) for agent in world.population]

    # Example: READ (Calculate) the number of agents at goal:
    num_at_goal = sum([int(world.goals[0].agent_achieved_goal(agent)) for agent in world.population])

    # print(positions[0])
    # print(num_at_goal)

    # Example: WRITE random controllers to all agents every frame
    for agent in world.population:
        agent.controller = get_random_controller()

    # Example: WRITE (Change) all agents to be a new shade of red every frame
    new_color = (int((random.random() * 200) + 55), 0, 0)
    for agent in world.population:
        agent.body_color = new_color


# Main Function
if __name__ == "__main__":
    robot_config = configure_robots(network)
    world_config = configure_env(robot_config=robot_config, num_agents=20)
    world_subscriber = WorldSubscriber(func=callback)

    # print(robot_config)

    simulator(
        show_gui=False,
        gui=TennlabGUI(),
        world_config=world_config,
        subscribers=[world_subscriber]
    )

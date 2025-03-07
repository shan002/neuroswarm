import novel_swarms

from novel_swarms.world.RectangularWorld import RectangularWorld, RectangularWorldConfig
from novel_swarms.agent.MazeAgent import MazeAgent, MazeAgentConfig
from novel_swarms.world.simulate import main as sim
from novel_swarms.agent.control.StaticController import StaticController
from novel_swarms.world.spawners.AgentSpawner import PointAgentSpawner
from novel_swarms.sensors.BinaryFOVSensor import BinaryFOVSensor
from novel_swarms.agent.control.BinaryController import BinaryController


# Adding world
world_config = RectangularWorldConfig(size=(10,10), time_step=1/40)
world = RectangularWorld(world_config)

# Adding an agent
agent_config = MazeAgentConfig(position=(5,5), agent_radius=0.1)
agent = MazeAgent(agent_config, world)
world.population.append(agent)


sensor = BinaryFOVSensor(agent, theta=0.45, distance=2) #theta in radians
agent.sensors.append(sensor)

# Adding a controller
# controller = StaticController(output=(0.01, 0.1)) # 10 cm/s forwards, 0.1 rad/s clockwise.
controller = BinaryController(agent, (0.02, -0.5), (0.02, 0.5))
agent.controller = controller

spawner = PointAgentSpawner(world, n=4, facing="away", avoid_overlap=True, agent=agent, oneshot=True)
world.spawners.append(spawner)

del world.population[:]
spawner.mark_for_deletion = False  # Re-enable the spawner
world.spawners.append(spawner)

sim(world)
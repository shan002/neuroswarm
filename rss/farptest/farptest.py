from pathlib import Path

from swarmsim.config import register_dictlike_type, register_agent_type
from swarmsim.agent.MazeAgent import MazeAgentConfig
from swarmsim.world.RectangularWorld import RectangularWorldConfig
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
from swarmsim.world.simulate import main as simulator
# from ..gui import TennlabGUI

cwd = Path(__file__).resolve().parent
config = RectangularWorldConfig.from_yaml(cwd / "world.yaml")

# gui = TennlabGUI(x=0, y=0, h=0, w=300)
# gui.position = "sidebar_right"

simargs = dict(
    world_config=config,
    subscribers=[],
    # gui=gui,
    show_gui=True,
    start_paused=True,
)
world = simulator(**simargs)  # run simulator

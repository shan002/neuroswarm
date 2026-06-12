import sys
from pathlib import Path

# Add proj-neuro to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'turtwig'))

import experiment_tenn2 as t2


class FARP1Experiment(t2.ConnorMillingExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    # def __init__(self, args):
    #     super(t2.ConnorMillingExperiment, self).__init__(args)
    #     self.world_yaml = args.world_yaml
    #     self.run_info = None

    #     self.track_history = args.track_history or args.log_trajectories
    #     self.log_trajectories = args.log_trajectories
    #     self.use_caspian = getattr(args, 'caspian', True)

    #     if self.agents is None and self.args.action != 'train':
    #         try:
    #             self.agents = self.p.experiment['agents']
    #         except (KeyError, IndexError, FileNotFoundError, AttributeError):
    #             pass

    #     self.start_paused = getattr(args, 'start_paused', False)

    #     self.n_inputs, self.n_outputs, _, _ = self.bootstrap_controller_encoders()

    #     self.log("initialized farp1")

    def simulate(self, processor, network, init_callback=None):
        from swarmsim.config import register_dictlike_type, register_agent_type
        from swarmsim.agent.MazeAgent import MazeAgentConfig
        from swarmsim.world.RectangularWorld import RectangularWorldConfig
        from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber as WorldSubscriber
        from swarmsim.world.simulate import main as simulator
        from rss.gui import TennlabGUI

        # setup network
        network.set_data("processor", self.processor_params)

        # register agent and controller type with RSS
        if self.use_caspian:
            register_dictlike_type('controller', "CaspianBinaryController", self.controller)
            register_dictlike_type('controller', "CaspianBinaryRemappedController", self.controller_remapped)
        else:
            register_dictlike_type('controller', "CaspianBinaryController", self.controller)
            register_dictlike_type('controller', "CaspianBinaryRemappedController", self.controller_remapped)

        # setup world
        config = RectangularWorldConfig.from_yaml(self.world_yaml)
        config.stop_at = self.cycles
        agent_config = config.spawners[0]['agent']
        agent_config['track_io'] = self.track_history
        controller_config = agent_config['controller']
        controller_config['neuro_track_all'] = self.viz
        controller_config['network'] = network
        if self.agents is not None:
            config.spawners[0]['n'] = int(self.agents)

        config.metrics = [
            # Coverage(history=float('inf'), regularize=True),
        ]

        def callback(world, screen):
            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.controller.neuron_ids,
                    "Event Counts": a.controller.neuron_counts
                })

        gui = TennlabGUI(x=0, y=0, h=0, w=300)
        gui.position = "sidebar_right"
        if self.viz is False or self.noviz:
            gui = False

        world_subscriber = WorldSubscriber(func=callback)

        simargs = dict(
            world_config=config,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
            start_paused=self.start_paused,
        )

        # allow for callback to modify config
        if (callable(init_callback)
            or hasattr(self, 'init_callback') and (init_callback := self.init_callback)):
            simargs = init_callback(self, simargs)

        world = simulator(**simargs)  # run simulator
        return world


def get_parsers(parser, subpar):
    return t2.get_parsers(parser, subpar)


if __name__ == "__main__":
    t2.main(name="farp1-v01",
            cls=FARP1Experiment,
            parser_callback=get_parsers)

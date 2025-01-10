# from multiprocessing import Pool, TimeoutError
# from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import random
# import os
# import time
# import pathlib
import numpy as np
# import matplotlib.pyplot as plt

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment

# from novel_swarms.agent.MazeAgentCaspian import MazeAgentCaspian
# from novel_swarms.agent.MillingAgentCaspian import MillingAgentCaspian


class NetworkCloneExperiment(TennExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    def __init__(self, args):
        super().__init__(args)
        self.agent_yaml = args.agent_yaml
        self.world_yaml = args.world_yaml
        self.run_info = None

        # self.n_inputs, self.n_outputs, self.encoder, self.decoder = (
        #     MillingAgentCaspian.get_default_encoders(self.app_params['proc_ticks']))

        # number of ticks to run the processor every control (sensor->actuator) cycle
        self.neuro_tpc = args.encoder_ticks

        self.n_inputs, self.n_outputs = (2, 2)
        encoder_params = {
            "dmin": [0] * self.n_inputs,  # two bins for each binary input + random
            "dmax": [1] * self.n_inputs,
            "interval": self.neuro_tpc,
            "named_encoders": {"s": "spikes"},
            "use_encoders": ["s"] * self.n_inputs
        }
        decoder_params = {
            # see notes near where decoder is used
            "dmin": [0] * self.n_outputs,
            "dmax": [1] * self.n_outputs,
            "divisor": self.neuro_tpc,
            "named_decoders": {"r": {"rate": {"discrete": False}}},
            "use_decoders": ["r"] * self.n_outputs
        }
        self.encoder = neuro.EncoderArray(encoder_params)
        self.decoder = neuro.DecoderArray(decoder_params)

        self.log("initialized experiment_tenn2iffclone")

    def setup_processor(self, network):
        # setup network; create, setup, & return processor
        # pprops = processor.get_configuration()
        processor = caspian.Processor(self.processor_params)
        processor.load_network(network)
        neuro.track_all_output_events(processor, network)  # track only output fires

        if self.viz:  # used for visualizing network activity
            neuro.track_all_neuron_events(processor, network)
            network.make_sorted_node_vector()
            self.neuron_ids = [x.id for x in network.sorted_node_vector]

        return processor

    @staticmethod
    def bool_to_one_hot(x: bool):
        return (0, 1) if x else (1, 0)

    def fitness(self, processor, network):
        b2oh = self.bool_to_one_hot

        def target_func(observation):
            if observation:
                v, w = 0.2, 0.3
            else:
                v, w = 0.8, 0.9

            return (v, w)
            # return 1.0 if observation else 0.0

        # setup sim

        # network.set_data("processor", self.processor_params)
        self.processor = self.setup_processor(network)

        error = 0
        for i in range(999):
            observation = random.getrandbits(1)  # fake observation
            target = self.target_func(observation)
            # input_vector = (1,)
            # target = 1
            # x = robot.run_processor(observation)

            # translate observation to vector
            input_vector = b2oh(observation)
            # input_vector += (1,)  # add 1 as constant on input to 4th input neuron

            spikes = self.encoder.get_spikes(input_vector)
            self.processor.apply_spikes(spikes)
            self.processor.run(5)
            self.processor.run(self.neuro_tpc)
            # action: bool = bool(proc.output_vectors())  # old. don't use.
            # if self.neuro_track_all:
            #    self.neuron_counts = self.processor.neuron_counts()
            data = self.decoder.get_data_from_processor(self.processor)

            x = data
            # x = data[0]

            error -= sum(np.power(np.subtract(x, target), 2))
            # error -= (x - target) ** 2

            if self.viz:
                print(f"{'see' if observation else 'not'}\ttruth: {target}\tproc:{x}\te:{error:.1}")

        return error


def get_parsers(parser, subpar):
    # this is a separate function so we can inherit options from this module
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        sub.add_argument('--agent_yaml', default="rss/turbopi-milling/turbopi.yaml",
                         type=str, help="path to yaml config for agent")
        sub.add_argument('--world_yaml', default="rss/turbopi-milling/world.yaml",
                         type=str, help="path to yaml config for world")
        sub.add_argument('', default="rss/turbopi-milling/world.yaml",
                         type=str, help="path to yaml config for world")
        sub.add_argument('--encoder_ticks', type=int, default=10,
                           help="SNN encoder ticks cycle. default=10")

    # Training args
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    return parser, subpar


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "fakesim_snn_eons-v01"
    if args.project is None and args.logfile is None:
        args.logfile = "tenn2symclone_train.log"


    app = ConnorMillingExperiment(args)

    # Do the appropriate action
    if args.action == "train":
        common.experiment.train(app, args)
    else:
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

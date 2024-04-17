import random

# Provided Python utilities from tennlab framework/examples/common
from common.experiment import TennExperiment
import common.experiment

try:
    import pandas
except ImportError:
    pandas = None


class TennBots(TennExperiments):
    """Tennbots application for TennLab neuro framework


    """

    def __init__(self, **kwargs):
        self.n_inputs = 2  # default. set one later in the code if necessary
        self.n_outputs = 2  # default. set one later in the code if necessary

        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # for now they are just being used to calculate n_inputs/outputs

        # Setup encoder
        # for each binary raw input, we encode it to constant spikes on bins, kinda like traditional one-hot
        encoder_params = {
            "dmin": [0] * 5,  # two bins for each binary input
            "dmax": [1] * 5,
            "interval": self.app_params['proc_ticks'],
            "named_encoders": {"s": "spikes"},
            "use_encoders": ["s"] * 5
        }
        self.encoder = neuro.EncoderArray(encoder_params)
        self.n_inputs = self.encoder.get_num_neurons()

        # Setup decoder
        # Read spikes to wheelspeeds using rate-based decoding
        decoder_params = {
            # see notes near where decoder is used
            "dmin": [0] * 4,
            "dmax": [1] * 4,
            "divisor": self.app_params['proc_ticks'],
            "named_decoders": {"r": {"rate": {"discrete": False}}},
            "use_decoders": ["r", "r", "r", "r"]
        }
        self.decoder = neuro.DecoderArray(decoder_params)
        self.n_outputs = self.decoder.get_num_neurons()

        # Set up the initial gym, and set up the action space.

        # env = gym.make(self.env_name)

        # self.action_min = [0]
        # self.action_max = [env.action_space.n-1]
        # self.action_type = env.action_space.dtype.type

        self.log("initialized experiment_tenn1")

        # If we're playing from stdin, we can return now -- nothing
        # else is being used

    def fitness(self, processor, network):
        return sum(self.run(processor, network) for i in range(self.runs))

    def run(self, processor, network, prerun_callback=lambda sim: None):
        import tennbots
        # setup sim
        rng = random.Random()
        sim = tennbots.Sim(self.agents, rng, render_mode="human" if self.viz else None)
        # setup processors
        pprops = processor.get_configuration()
        # print(pprops)
        processors = [caspian.Processor(pprops)] * self.agents
        for proc in processors:
            proc.load_network(network)
            neuro.track_all_output_events(proc, network)
            if self.iostream is not None:
                neuro.track_all_neuron_events(proc, network)
        # setup a spot to save actions to
        actions = [(0, 0)] * self.agents
        loss_graph = []

        if self.iostream is not None:
            network.make_sorted_node_vector()
            ids = [x.id for x in network.sorted_node_vector]

        # save references for loop optimization
        encoder, decoder = self.encoder, self.decoder
        proc_ticks = self.app_params['proc_ticks']

        def b2oh(x: bool):
            return (0, 1) if x else (1, 0)

        def get_action(processor, observation, i=None):
            # unpack observation
            ir_sensed, goal_sensed = observation
            # convert each binary to one-hot and concatenate
            # input_vector = b2oh(ir_sensed) + b2oh(goal_sensed) + (1, )
            input_vector = b2oh(ir_sensed) + b2oh(goal_sensed) + (rng.randint(0, 1), )
            spikes = encoder.get_spikes(input_vector)
            processor.apply_spikes(spikes)
            processor.run(10)
            processor.run(proc_ticks)
            # action: bool = bool(proc.output_vectors())  # old. don't use.
            if self.iostream is not None and i == 0:
                # print(i, processor.neuron_counts())
                rv = {"Neuron Alias": ids, "Event Counts": processor.neuron_counts()}
                self.iostream.write_json(rv)
            data = decoder.get_data_from_processor(processor)
            # four bins. Two for each parameter.
            v = 0.2 * (data[1] - data[0])  # m/s
            w = 2.0 * (data[3] - data[2])  # rad/s
            return (v, w)

        prerun_callback(sim)

        for i in range(self.sim_time):
            observations, reward, _, _, info, *_ = sim.step(actions)
            loss_graph.append(reward)
            actions = (get_action(p, o, i) for i, (p, o) in enumerate(zip(processors, observations)))
            # print(f"obsv: {observations}\n\n")
            # print(f"act: {actions}\n\n")
            if self.viz:
                sim.render()

        observations, reward, _, _, info, *_ = sim.step(actions)
        # how_many_goal_sensed = sum([goal_sensed for ir_sensed, goal_sensed in observations])

        # reward += (how_many_on_goal ** 2) * 1000

        # loss = sum(loss_graph[-5000:])
        return reward


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "ksim_snn_eons-goalsearch-v01"

    app = TennBots(args)

    # Do the appropriate action
    if args.action == "train":
        common.experiment.train(app, args)
    else:
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

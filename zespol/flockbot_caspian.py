import neuro
import caspian

from flockbot_binarycontroller import FlockbotBinarycontroller

# typing
from typing import List, Tuple
try:
    from swarms.lib.sensors.binary import BinarySensor
except (ImportError, ModuleNotFoundError):
    pass


class FlockbotCaspian(FlockbotBinarycontroller):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        heading: float,
        spt: float,
        sensors: List[BinarySensor],
        network: dict,
        agent_radius: float = 0.16,
        neuro_track_all: bool = False,
    ) -> None:
        super().__init__(pos=pos, heading=heading, sensors=sensors,
                         name=name, controller="Caspian", spt=spt)

        self.network = network

        # for tracking neuron activity
        self.neuron_counts = None
        self.neuron_ids = None
        self.neuro_track_all = neuro_track_all

        self.setup_encoders()

        self.processor_params = self.network.get_data("processor")
        if self.network is not None:
            self.setup_processor(self.processor_params)

    @staticmethod  # to get encoder structure/#neurons for external network generation (EONS)
    def get_default_encoders(neuro_tpc):
        encoder_params = {
            "dmin": [0] * 2,  # two bins for each binary input + extra
            "dmax": [1] * 2,
            "interval": neuro_tpc,
            "named_encoders": {"s": "spikes"},
            "use_encoders": ["s"] * 2
        }
        decoder_params = {
            # see notes near where decoder is used
            "dmin": [0] * 4,
            "dmax": [1] * 4,
            "divisor": neuro_tpc,
            "named_decoders": {"r": {"rate": {"discrete": True}}},
            "use_decoders": ["r"] * 4
        }
        encoder = neuro.EncoderArray(encoder_params)
        decoder = neuro.DecoderArray(decoder_params)

        return (
            encoder.get_num_neurons(),
            decoder.get_num_neurons(),
            encoder,
            decoder
        )

    def setup_encoders(self, class_homogenous=True) -> None:
        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # Setup encoder
        # for each binary raw input, we encode it to constant spikes on bins, kinda like traditional one-hot

        # Setup decoder
        # Read spikes to a discrete set of floats using rate-based decoding

        x = FlockbotCaspian if class_homogenous else self

        encoders = x.get_default_encoders(self.spt)

        x.n_inputs, x.n_outputs, x.encoder, x.decoder = encoders

    def setup_processor(self, pprops):
        # pprops = processor.get_configuration()
        self.processor = caspian.Processor(pprops)
        self.processor.load_network(self.network)
        neuro.track_all_output_events(self.processor, self.network)  # track only output fires

        if self.neuro_track_all:  # used for visualizing network activity
            neuro.track_all_neuron_events(self.processor, self.network)
            self.network.make_sorted_node_vector()
            self.neuron_ids = [x.id for x in self.network.sorted_node_vector]

    @staticmethod
    def bool_to_one_hot(x: bool):
        return (0, 1) if x else (1, 0)

    def run_processor(self, observations):
        b2oh = self.bool_to_one_hot

        # translate observation to vector
        input_vector = b2oh(observations[0])
        input_vector += (1,)  # add 1 as constant on input to 5th input neuron
        # input_vector += (self.rng.randint(0, 1),)  # add random input to 5th input neuron

        spikes = self.encoder.get_spikes(input_vector)
        self.processor.apply_spikes(spikes)
        self.processor.run(self.neuro_tpc)
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()
        data = self.decoder.get_data_from_processor(self.processor)
        """  old wheelspeed code.
            # four bins. Two for each wheel, one for positive, one for negative.
            wl, wr = 2 * (data[1] - data[0]), 2 * (data[3] - data[2])
            return (wl, wr)
        """
        # three bins. One for +v, -v, omega.
        v = (data[1] - data[0])
        w = (data[3] - data[2])
        return v, w

    def get_action(self, world_state) -> Tuple:
        # observations = (sensor.sense() for sensor in self.sensors)
        observations = [self.sensor.sense(
            agent_name=self.name,
            world_state=world_state
        )]

        a, b = self.run_processor(observations)
        v = a * 0.2
        omega = b * 2.0
        self.requested = v, omega
        return self.requested

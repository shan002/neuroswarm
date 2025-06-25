import math
import numpy as np

# typing
from typing import Any, override

from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.AbstractController import AbstractController

import casPYan
import casPYan.ende.rate as ende


class CasPyanBinaryController(AbstractController):

    def __init__(
        self,
        agent,
        parent=None,
        network: dict[str, Any] | None = None,
        neuro_tpc: int | None = 10,
        extra_ticks: int = 5,
        neuro_track_all: bool = False,
        scale_forward_speed: float = 0.2,  # m/s forward speed factor
        scale_turning_rates: float = 2.0,  # rad/s turning rate factor
        sensor_id: int = 0,
    ) -> None:
        # if config is None:
        #     config = MazeAgentCaspianConfig()

        super().__init__(agent=agent, parent=parent)

        self.scale_v = scale_forward_speed  # m/s
        self.scale_w = scale_turning_rates  # rad/s

        self.network = network

        # for tracking neuron activity
        self.neuron_counts = None
        self.neuron_ids = None
        self.neuro_track_all = neuro_track_all

        # how many ticks the neuromorphic processor should run for
        if neuro_tpc is None:
            try:
                app_params = self.network.get_data("application")
                self.neuro_tpc = app_params["encoder_ticks"]
            except RuntimeError as err:
                raise RuntimeError("Could not find application parameters in network and no neuro_tpc specified.") from err
        else:
            self.neuro_tpc = neuro_tpc
        # now that we have the ticks per processor cycle, we can setup encoders & decoders
        self.setup_encoders()

        self.extra_ticks = extra_ticks

        self.processor_params = self.network.get_data("processor")
        self.setup_processor(self.processor_params)

        self.sensor_id = sensor_id

        # typing
        self.n_inputs: int
        self.n_outputs: int
        self.encoder: list
        self.decoder: list
        self.processor: casPYan.Processor

    @staticmethod  # to get encoder structure/#neurons for external network generation (EONS)
    def get_default_encoders(neuro_tpc=1):
        encoder_neurons, decoder_neurons = 2, 4
        encoders = [ende.RateEncoder(neuro_tpc, [0.0, 1.0]) for _ in range(encoder_neurons)]
        decoders = [ende.RateDecoder(neuro_tpc, [0.0, 1.0]) for _ in range(decoder_neurons)]

        return (
            len(encoders),
            len(decoders),
            encoders,
            decoders
        )

    def setup_encoders(self) -> None:
        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # Setup encoder
        # for each binary raw input, we encode it to constant spikes on bins, kinda like traditional one-hot
        # Setup decoder
        # Read spikes to a discrete set of floats using rate-based decoding
        x = self
        encoders = x.get_default_encoders(self.neuro_tpc)
        x.n_inputs, x.n_outputs, x.encoder, x.decoder = encoders

    def setup_processor(self, pprops):
        # pprops = processor.get_configuration()
        self.processor = casPYan.Processor(pprops)
        self.processor.load_network(self.network)
        # neuro.track_all_output_events(self.processor, self.network)  # track only output fires

        if self.neuro_track_all:  # used for visualizing network activity
            # neuro.track_all_neuron_events(self.processor, self.network)
            # self.network.make_sorted_node_vector()
            self.neuron_ids = self.processor.names

    @staticmethod
    def bool_to_one_hot(x: bool):
        return (0, 1) if x else (1, 0)

    def run_processor(self, observation):
        b2oh = self.bool_to_one_hot

        input_vector = b2oh(observation)

        # encode to spikes
        input_slice = input_vector[:len(self.encoder)]
        spikes = [enc.get_spikes(x) for enc, x in zip(self.encoder, input_slice)]
        # run processor
        self.processor.apply_spikes(spikes)
        self.processor.run(self.extra_ticks)
        self.processor.run(self.neuro_tpc)
        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()
        data = [dec.decode(node.history) for dec, node in zip(self.decoder, self.processor.outputs)]
        """  old wheelspeed code.
            # four bins. Two for each wheel, one for positive, one for negative.
            wl, wr = 2 * (data[1] - data[0]), 2 * (data[3] - data[2])
            return (wl, wr)
        """
        # three bins. One for +v, -v, omega.
        v = self.scale_v * (data[1] - data[0])
        w = self.scale_w * (data[3] - data[2])
        return v, w

    def get_actions(self, agent) -> tuple[float, float]:
        sensor: BinaryFOVSensor = self.parent.sensors[0]
        self.parent.set_color_by_id(sensor.detection_id)

        v, omega = self.run_processor(sensor.current_state)
        self.requested = v, omega
        return self.requested

# CaspianTernaryController.py

import math
import numpy as np

# typing
from typing import Any, override

# Import the new TernaryFOVSensor instead of BinaryFOVSensor
from rss.TernaryFOVSensor import TernaryFOVSensor
from novel_swarms.agent.control.AbstractController import AbstractController

import neuro
import caspian


class CaspianTernaryController(AbstractController):

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
        super().__init__(agent=agent, parent=parent)

        self.scale_v = scale_forward_speed  # m/s
        self.scale_w = scale_turning_rates  # rad/s

        self.network: neuro.Network = network if network is not None else network

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
                raise RuntimeError(
                    "Could not find application parameters in network and no neuro_tpc specified."
                ) from err
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
        self.encoder: neuro.EncoderArray
        self.decoder: neuro.DecoderArray
        self.processor: caspian.Processor

    @staticmethod  # to get encoder structure/#neurons for external network generation (EONS)
    def get_default_encoders(neuro_tpc=1):
        # Now we have three possible sensor states: 0, 1, or 2
        encoder_neurons, decoder_neurons = 3, 4
        encoder_params = {
            "dmin": [0] * encoder_neurons,
            "dmax": [1] * encoder_neurons,
            "interval": neuro_tpc,
            "named_encoders": {"s": "spikes"},
            "use_encoders": ["s"] * encoder_neurons,
        }
        decoder_params = {
            "dmin": [0] * decoder_neurons,
            "dmax": [1] * decoder_neurons,
            "divisor": neuro_tpc,
            "named_decoders": {"r": {"rate": {"discrete": False}}},
            "use_decoders": ["r"] * decoder_neurons,
        }
        encoder = neuro.EncoderArray(encoder_params)
        decoder = neuro.DecoderArray(decoder_params)

        return (
            encoder.get_num_neurons(),
            decoder.get_num_neurons(),
            encoder,
            decoder,
        )

    def setup_encoders(self) -> None:
        # Setup encoder & decoder based on the changed neuron counts
        encoders = self.get_default_encoders(self.neuro_tpc)
        self.n_inputs, self.n_outputs, self.encoder, self.decoder = encoders

    def setup_processor(self, pprops):
        self.processor = caspian.Processor(pprops)
        self.processor.load_network(self.network)
        neuro.track_all_output_events(self.processor, self.network)  # track only output fires

        if self.neuro_track_all:  # used for visualizing network activity
            neuro.track_all_neuron_events(self.processor, self.network)
            self.network.make_sorted_node_vector()
            self.neuron_ids = [x.id for x in self.network.sorted_node_vector]

    @staticmethod
    def int_to_one_hot(h: int):
        """
        Convert a ternary h ∈ {0, 1, 2} into a one‐hot vector of length 3:
          h = 0 → (1, 0, 0)
          h = 1 → (0, 1, 0)
          h = 2 → (0, 0, 1)
        """
        if h == 0:
            return (1, 0, 0)
        elif h == 1:
            return (0, 1, 0)
        else:
            return (0, 0, 1)

    def run_processor(self, observation: int):
        # observation now is an integer 0, 1, or 2 from TernaryFOVSensor.current_state
        input_vector = self.int_to_one_hot(observation)
        spikes = self.encoder.get_spikes(input_vector)

        self.processor.apply_spikes(spikes)
        self.processor.run(self.extra_ticks)
        self.processor.run(self.neuro_tpc)

        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()

        data = self.decoder.get_data_from_processor(self.processor)
        # same decoder logic as before: three bins for +v, -v, ω
        v = self.scale_v * (data[1] - data[0])
        w = self.scale_w * (data[3] - data[2])
        return v, w

    def get_actions(self, agent) -> tuple[float, float]:
        # Use TernaryFOVSensor instead of BinaryFOVSensor
        sensor: TernaryFOVSensor = self.parent.sensors[self.sensor_id]
        self.parent.set_color_by_id(sensor.detection_id)

        # Pass the integer current_state (0,1,2) into run_processor
        v, omega = self.run_processor(sensor.current_state)
        self.requested = v, omega
        return self.requested

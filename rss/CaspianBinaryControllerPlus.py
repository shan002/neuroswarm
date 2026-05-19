import math
import numpy as np

# typing
from typing import Any, override

from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.AbstractController import AbstractController

import neuro
import caspian


class CaspianBinaryControllerPlus(AbstractController):

	def __init__(
		self,
		agent,
		parent=None,
		network: dict[str, Any] | None = None,
		neuro_tpc: int | None = 10,
		extra_ticks: int = 5,
		neuro_track_all: bool = False,
		scale_forward_speed: float = 0.276,  # m/s forward speed factor
		scale_turning_rates: float = 0.602,  # rad/s turning rate factor
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
		self.encoder: neuro.EncoderArray
		self.decoder: neuro.DecoderArray
		self.processor: caspian.Processor

	@staticmethod  # to get encoder structure/#neurons for external network generation (EONS)
	def get_default_encoders(neuro_tpc=1):
		encoder_neurons, decoder_neurons = 2, 4
		encoder_params = {
			"dmin": [0] * encoder_neurons,  # two bins for each binary input + random
			"dmax": [1] * encoder_neurons,
			"interval": neuro_tpc,
			"named_encoders": {"s": "spikes"},
			"use_encoders": ["s"] * encoder_neurons
		}
		decoder_params = {
			# see notes near where decoder is used
			"dmin": [0] * decoder_neurons,
			"dmax": [1] * decoder_neurons,
			"divisor": neuro_tpc,
			"named_decoders": {"r": {"rate": {"discrete": False}}},
			"use_decoders": ["r"] * decoder_neurons
		}
		encoder = neuro.EncoderArray(encoder_params)
		decoder = neuro.DecoderArray(decoder_params)

		return (
			encoder.get_num_neurons(),
			decoder.get_num_neurons(),
			encoder,
			decoder
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

	@staticmethod
	def quantize_unit_interval(x: float, steps: int) -> float:
		if steps <= 1:
			return 0.0
		if x < 0.0:
			x = 0.0
		elif x > 1.0:
			x = 1.0
		return round(x * (steps - 1)) / (steps - 1)

	def run_processor(self, observation):
		b2oh = self.bool_to_one_hot

		input_vector = b2oh(observation)

		spikes = self.encoder.get_spikes(input_vector)
		self.processor.apply_spikes(spikes)
		self.processor.run(self.extra_ticks)
		if self.neuro_track_all:
			neuron_counts = np.asarray(self.processor.neuron_counts())
		self.processor.run(self.neuro_tpc)
		if self.neuro_track_all:
			neuron_counts += self.processor.neuron_counts()
			self.neuron_counts = neuron_counts.tolist()
		data = self.decoder.get_data_from_processor(self.processor)

		# Quantize to more than 3 steps in [0, 1] before scaling to speeds.
		steps = 5
		data = [self.quantize_unit_interval(x, steps) for x in data]

		v = self.scale_v * (data[1] - data[0])
		w = self.scale_w * (data[3] - data[2])
		return v, w

	def get_actions(self, agent) -> tuple[float, float]:
		sensor: BinaryFOVSensor = self.parent.sensors[0]
		self.parent.set_color_by_id(sensor.detection_id)

		v, omega = self.run_processor(sensor.current_state)
		self.requested = v, omega
		return self.requested

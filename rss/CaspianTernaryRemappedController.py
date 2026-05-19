from .CaspianTernaryController import CaspianTernaryController
from typing import override

import neuro
import caspian


class CaspianTernaryRemappedController(CaspianTernaryController):

    @override
    def run_processor(self, observation):
        # Convert ternary observation to one-hot input
        input_vector = self.one_hot(observation)
        spikes = self.encoder.get_spikes(input_vector)

        # Apply spikes and run processor
        self.processor.apply_spikes(spikes)
        self.processor.run(5)
        self.processor.run(self.neuro_tpc)

        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()

        # Decode neuron outputs
        data = self.decoder.get_data_from_processor(self.processor)
        data = [int(round(x)) for x in data]

        # Remapped speed and turn mappings
        v_mapping = [0.0, self.scale_v]
        w_mapping = [0.0, self.scale_w]

        # Compute forward speed and turning rate
        v = v_mapping[data[1]] - v_mapping[data[0]]
        w = w_mapping[data[3]] - w_mapping[data[2]]

        # If zero, randomize direction
        if v == 0.0:
            v = v_mapping[1] * self.parent.rng.choice([-1, 1])
        if w == 0.0:
            w = w_mapping[1] * self.parent.rng.choice([-1, 1])

        return v, w

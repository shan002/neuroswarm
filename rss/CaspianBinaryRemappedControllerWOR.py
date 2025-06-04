from .CaspianBinaryController import CaspianBinaryController

# typing
from typing import Any, override

import neuro
import caspian


class CaspianBinaryRemappedControllerWOR(CaspianBinaryController):

    @override
    def run_processor(self, observation):
        b2oh = self.bool_to_one_hot

        # translate observation to vector
        input_vector = b2oh(observation)
        # input_vector += (1,)  # add 1 as constant on input to 4th input neuron

        spikes = self.encoder.get_spikes(input_vector)
        self.processor.apply_spikes(spikes)
        self.processor.run(5)
        self.processor.run(self.neuro_tpc)
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()
        data = self.decoder.get_data_from_processor(self.processor)
        data = [int(round(x)) for x in data]

        v_mapping = [0.0, 0.276,]
        w_mapping = [0.0, 0.602,]

        v = v_mapping[data[1]] - v_mapping[data[0]]
        w = w_mapping[data[3]] - w_mapping[data[2]]

        return v, w
        # return (0.08, 0.4) if not observation else (0.18, 0.0)  # CMA best controller

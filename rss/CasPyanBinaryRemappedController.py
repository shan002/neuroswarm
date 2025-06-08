from .CasPyanBinaryController import CasPyanBinaryController

# typing
from typing import Any, override

import neuro
import caspian


class CasPyanBinaryRemappedController(CasPyanBinaryController):

    @override
    def run_processor(self, observation):
        b2oh = self.bool_to_one_hot

        # translate observation to vector
        input_vector = b2oh(observation)
        # input_vector += (1,)  # add 1 as constant on input to 4th input neuron

        # encode to spikes
        input_slice = input_vector[:len(self.encoder)]
        spikes = [enc.get_spikes(x) for enc, x in zip(self.encoder, input_slice)]
        # run processor
        self.processor.apply_spikes(spikes)
        self.processor.run(5)
        self.processor.run(self.neuro_tpc)
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        if self.neuro_track_all:
            self.neuron_counts = self.processor.neuron_counts()
        data = [dec.decode(node.history) for dec, node in zip(self.decoder, self.processor.outputs)]
        data = [int(round(x)) for x in data]
        # three bins. One for +v, -v, omega.
        # v = self.scale_v * (data[1] - data[0])
        # w = self.scale_w * (data[3] - data[2])
        # these values were taken from an average of speeds/turning rates
        # from measurements of Turbopis 1, 2, 3, 4 @ (100, 90, +-0.5)
        v_mapping = [0.0, 0.276,]
        w_mapping = [0.0, 0.602,]
        v = v_mapping[data[1]] - v_mapping[data[0]]
        w = w_mapping[data[3]] - w_mapping[data[2]]
        if v == 0.0:
            v = v_mapping[1] * self.parent.rng.choice([-1, 1])
            # print(1 if v > 0 else 0)
        if w == 0.0:
            w = w_mapping[1] * self.parent.rng.choice([-1, 1])

        return v, w
        # return (0.08, 0.4) if not observation else (0.18, 0.0)  # CMA best controller

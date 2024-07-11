import neuro
import caspian
import risp
import json

import tabulate

caspian_config = {
    "Leak_Enable": True,
    "Min_Leak": -1,
    "Max_Leak": 4,
    "Min_Axon_Delay": 0,
    "Max_Axon_Delay": 0,
    "Min_Threshold": 0,
    "Max_Threshold": 127,
    "Min_Synapse_Delay": 0,
    "Max_Synapse_Delay": 1000,
    "Min_Weight": -127,
    "Max_Weight": 127
}

Cpu = caspian.Processor

npu = Cpu(caspian_config)

net = neuro.Network()
# with open("experiment_tenn2_tenngineered-milling.json") as f:
#     j = json.loads(f.read())
# net.from_json(j)
# processor_params = net.get_data("processor")
# app_params = net.get_data("application").to_python()
# net.set_properties(proc.get_network_properties())
# print(net)

npu.load_network(net)
neuro.track_all_output_events(npu, net)

neuro.track_all_neuron_events(npu, net)
net.make_sorted_node_vector()
neuron_ids = [x.id for x in net.sorted_node_vector]

neuro_tpc = app_params['proc_ticks']

# encoder_params = {
#     "dmin": [0] * 2,  # two bins for each binary input + random
#     "dmax": [1] * 2,
#     "interval": neuro_tpc,
#     "named_encoders": {"s": "spikes"},
#     "use_encoders": ["s"] * 2
# }
# decoder_params = {
#     # see notes near where decoder is used
#     "dmin": [0] * 4,
#     "dmax": [1] * 4,
#     "divisor": neuro_tpc,
#     "named_decoders": {"r": {"rate": {"discrete": True}}},
#     "use_decoders": ["r"] * 4
# }
# encoder = neuro.EncoderArray(encoder_params)
# decoder = neuro.DecoderArray(decoder_params)

spikes = []

def run_processor(spikes):
    # spikes = encoder.get_spikes(vector)
    npu.apply_spikes(spikes)
    npu.run(neuro_tpc)

    neuron_counts = npu.neuron_counts()
    data = decoder.get_data_from_processor(npu)
    return neuron_counts, data

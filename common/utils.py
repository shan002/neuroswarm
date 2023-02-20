import math
import random
import neuro
import json

# Takes a string.  If the string has "{", it loads it as json.
# Otherwise, if the string has ".", then it treats it as a filename,
# and loads the json from the file.
# Otherwise, it simply returns the string to use as simply json.

def load_json_string_file(sorf):
    if (sorf.find("{") != -1): return json.loads(sorf)
    if (sorf.find("[") != -1): return json.loads(sorf)
    if (sorf.find(".") == -1): return sorf
    with open(sorf, 'r') as f: return json.loads(f.read())

# Creates a template network given inputs, outputs, and a Processor
def make_template(dev, n_inputs, n_outputs, moa = None):
    # Create a network suitable for the given processor
    net = neuro.Network()
    props = dev.get_network_properties()
    net.set_properties(props)
    tprop = net.get_node_property("Threshold")

    if moa is None:
        moa = neuro.MOA()
        moa.seed(random.randint(0, (2**32)-1))

    # Add inputs
    for i in range(n_inputs):
        node = net.add_node(i)
        net.add_input(i)
        net.randomize_node_properties(moa, node)
        node.set(tprop.index, tprop.min_value)

    # Add outputs
    for i in range(n_outputs):
        node = net.add_node(n_inputs+i)
        net.add_output(n_inputs+i)
        net.randomize_node_properties(moa, node)
        node.set(tprop.index, tprop.min_value)

    return net

def make_random_net(net, moa = None):

    # Ensure we have a template graph
    if net.num_inputs() == 0 or net.num_outputs() == 0:
        raise ValueError("random_network requires a template graph to be given") 

    # Get an RNG if one isn't given
    if moa is None:
        moa = neuro.MOA()
        moa.seed(random.randint(0, (2**32)-1))

    # Determine how many hidden nodes
    num_hidden = min(100, max(net.num_inputs(), net.num_outputs()))

    # TODO: parameterize
    i_to_h = int(min(20, max(2, 0.5 * num_hidden)))
    h_to_h = int(min(20, max(2, 0.5 * num_hidden)))
    h_to_o = int(max(5, 0.5 * num_hidden))

    # Generate a list of ids corresponding to the new hidden nodes
    hidden_nodes = [n for n in range(net.num_nodes(), net.num_nodes() + num_hidden)]

    #print("hidden: {} i->h: {} h->h: {} h->o: {}".format(num_hidden, i_to_h, h_to_h, h_to_o))

    # Add hidden nodes
    for nh in hidden_nodes:
        net.randomize_node_properties(moa, net.add_node(nh))

    ## Input -> Hidden
    for ni in range(net.num_inputs()):
        input_node = net.get_input(ni)
        net.randomize_node_properties(moa, input_node)

        #ih = random.randint(i_to_h // 2, 3 * i_to_h // 2)
        ih = i_to_h

        for _ in range(ih):
            h = random.randint(0, len(hidden_nodes)-1)
            edge = net.add_or_get_edge(input_node.id, hidden_nodes[h])
            net.randomize_edge_properties(moa, edge)

    ## Hidden -> Hidden
    for _ in range(h_to_h):
        random.shuffle(hidden_nodes)
        for a, b in zip(hidden_nodes, hidden_nodes[::-1]):
            edge = net.add_or_get_edge(a, b)
            net.randomize_edge_properties(moa, edge)

    ## Hidden -> Output
    for no in range(net.num_outputs()):
        output_node = net.get_node(no)
        net.randomize_node_properties(moa, output_node)

        #ih = random.randint(h_to_o // 2, 3 * h_to_o // 2)
        ih = h_to_o

        for _ in range(ih):
            h = random.randint(0, len(hidden_nodes)-1)
            edge = net.add_or_get_edge(output_node.id, hidden_nodes[h])
            net.randomize_edge_properties(moa, edge)

# Create a feed-forward network, given the processor, an array describing the size of the layers, and the random number generator 
def make_feed_forward(dev, layer_size, moa):
    net = neuro.Network()
    props = dev.get_properties()
    net.set_properties(props)
    if (len(layer_size) < 2):
        raise ValueError("make_feed_forward requires layers to have at least an input layer and an output layer")

    num_inputs = layer_size[0]
    num_outputs = layer_size[-1]

    layers = []

    layers.append([])
    for i in range(num_inputs):
        node = net.add_node(i)
        net.randomize_node_properties(moa, node)
        net.add_input(i)
        layers[0].append(i)

    current_index = num_inputs+num_outputs

    for i in range(1,len(layer_size)-1):
        layers.append([])
        for j in range(layer_size[i]):
            node = net.add_node(current_index)
            net.randomize_node_properties(moa, node)
            layers[-1].append(current_index)
            current_index += 1

    layers.append([])
    for i in range(num_outputs):
        node = net.add_node(num_inputs+i)
        net.randomize_node_properties(moa, node)
        net.add_output(num_inputs+i)
        layers[-1].append(num_inputs+i)

    for i in range(len(layers)-1):
        for j in range(len(layers[i])):
            for k in range(len(layers[i+1])):
                edge = net.add_edge(layers[i][j], layers[i+1][k])
                net.randomize_edge_properties(moa, edge)
    return net   


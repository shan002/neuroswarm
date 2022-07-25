from multiprocessing import Pool, TimeoutError
import neuro
import eons
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', "--threads", default=4, type=int)
args = parser.parse_args()


props = neuro.PropertyPack()
props.add_node_property("x1", 1, 180, neuro.PropertyType.Integer)
props.add_node_property("x2", .1, 300)  # range
props.add_node_property("x3", .1, 20)  # turning_rate
props.add_node_property("x4", .1, 30)  # speed

template_genome = neuro.Network()
template_genome.set_properties(props)


node = template_genome.add_node(0)
node.set("x1", int(random.random() * 180))
node.set("x2", random.random() * 300)
node.set("x3", random.random() * 20)
node.set("x4", random.random() * 30)
template_genome.add_input(0)

eons_params = {
    "merge_rate": 0,
    "population_size": 100,
    "multi_edges": 0,
    "crossover_rate": 0.5,
    "mutation_rate": 0.5,
    "starting_nodes": 0,
    "starting_edges": 0,
    "selection_type": "tournament",
    "tournament_size_factor": 0.1,
    "tournament_best_net_factor": 0.9,
    "random_factor": 0.1,
    "num_best": 10,
    "num_mutations": 1,
    "node_mutations": {
        "x1": 1.0,
        "x2": 1.0,
        "x3": 1.0,
        "x4": 1.0,
    },
    "net_mutations": {},
    "edge_mutations": {},
    "add_node_rate": 0,
    "add_edge_rate": 0,
    "delete_node_rate": 0,
    "delete_edge_rate": 0,
    "net_params_rate": 0,
    "node_params_rate": 1,
    "edge_params_rate": 0
}

evolver = eons.EONS(eons_params)
evolver.set_template_network(template_genome)

pop = evolver.generate_population(eons_params, 1)


def fitness(props):
    import ricky
    global runsi
    # print(*props, end='\t')
    r = ricky.Run(10, *props, visible=False)
    for i in range(9999):
        r.step()
    loss = r.circleness()
    # print(loss)
    ricky.turtle.getscreen().bye()
    print('▬', end='', flush=True)
    return loss


def run_batch(networks):
    # print(len(networks))
    # print(networks[0])

    pop_props = []
    # TODO: use toJSON to send the full NetworkInfo over (for neuromorphic)
    for g in networks:  # unpack all values from NetworkInfo objects since those are unpickleable
        pop_props.append((
            g.network.get_node(0).get("x1"),
            g.network.get_node(0).get("x2"),
            g.network.get_node(0).get("x3"),
            g.network.get_node(0).get("x4"),
        ))

    # print(dill.detect.baditems(networks))
    with Pool(processes=args.threads) as pool:
        results = pool.map(fitness, pop_props)
    return results


print("Running", eons_params["population_size"])

vals = []
for i in range(500):
    print(f"[{'-' * 100}]", end='', flush=True)
    print('\b' * 101, end='', flush=True)    # Calculate the fitnesses of all of the networks in the population
    fitnesses = run_batch(pop.networks)
    # fitnesses = [fitness(g.network) for g in pop.networks]
    # Print information about the best network
    print('\033[2K\033[1G', end='', flush=True)
    max_fit = max(fitnesses)
    vals.append(max_fit)
    index = fitnesses.index(max_fit)
    gmax = pop.networks[index].network
    if (i % 1 == 0):
        print("Epoch\t{:4d}, Best Score: {}".format(i, max_fit))
        print("\t   FOV: {0:3.0f}\tRange: {1:6.3f}\tTurning: {2:6.3f}\tSpeed: {3:6.3f}\v".format(
            gmax.get_node(0).get("x1"),
            gmax.get_node(0).get("x2"),
            gmax.get_node(0).get("x3"),
            gmax.get_node(0).get("x4"),
        ))

    # Create the next population based on the fitnesses of the current population
    pop = evolver.do_epoch(pop, fitnesses, eons_params)

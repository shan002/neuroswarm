import asyncio
import turtwig.ricky as ricky
import neuro
import eons
import random
import matplotlib.pyplot as plt

props = neuro.PropertyPack()
props.add_node_property("value1", .1, 20)
props.add_node_property("value2", .1, 20)
# props.add_node_property("value2", 3, 30, neuro.PropertyType.Integer)

template_genome = neuro.Network()
template_genome.set_properties(props)


node = template_genome.add_node(0)
node.set("value1", random.random()*10)
node.set("value2", 4)
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
    "node_mutations": {"value1": 1.0, "value2": 1.0},
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


runsi = 0


def fitness(g):
    global runsi
    turning_rate = g.get_node(0).get("value2")
    speed = g.get_node(0).get("value1")
    print(runsi, turning_rate, speed, end='\t')
    r = ricky.Run(n=10, turning_rate=turning_rate, speed=speed, visible=False)
    for i in range(9999):
        r.step()
    loss = r.circleness()
    print(loss)
    runsi += 1
    ricky.turtle.getscreen().bye()
    return loss


async def run_batch(networks):
    print(len(networks))
    result = await asyncio.gather(*[fitness(g.network) for g in networks])
    ricky.turtle.getscreen().bye()
    return result


vals = []
for i in range(500):
    # Calculate the fitnesses of all of the networks in the population
    # fitnesses = asyncio.run(run_batch(pop.networks))
    fitnesses = [fitness(g.network) for g in pop.networks]
    # Print information about the best network
    max_fit = max(fitnesses)
    vals.append(max_fit)
    index = fitnesses.index(max_fit)
    gmax = pop.networks[index].network
    if (i % 50 == 0):
        print("Epoch {:4d}: {}".format(i, max_fit))
        print(gmax.get_node(0).get("value1"), ",", gmax.get_node(0).get("value2"))

    # Create the next population based on the fitnesses of the current population
    pop = evolver.do_epoch(pop, fitnesses, eons_params)

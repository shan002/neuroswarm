from multiprocessing import Pool, TimeoutError
from tqdm.contrib.concurrent import process_map
import sys
import neuro
import eons
import random
# import matplotlib.pyplot as plt
import argparse
import os
import time

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
    import ricky as sim
    global runsi
    # print(*props, end='\t')
    r = sim.Run(10, *props, visible=False)
    graph = []
    for i in range(9999):
        r.step()
        graph.append(r.circleness())
    # print(loss)
    loss = sum(graph[-5000:])
    sim.turtle.getscreen().bye()
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
    # with Pool(processes=args.threads) as pool:
    #     results = pool.map(fitness, pop_props)
    return process_map(fitness, pop_props, max_workers=args.threads)


print("Running", eons_params["population_size"])

vals = []


def do_epoch(pop):
    str1 = f"Epoch {i:4d}"
    print(str1)

    start_time = time.time()
    fitnesses = run_batch(pop.networks)    # Calculate the fitnesses of all of the networks in the population

    # https://en.wikipedia.org/wiki/ANSI_escape_code
    sys.stdout.write(f"\033[1F\033[2K\033[1G")  # Cursor up one line, clear line, cursor to beginning of line
    sys.stdout.write(f"\033[1F\033[2K\033[1G")  # Cursor up one line, clear line, cursor to beginning of line
    elapsed = time.time() - start_time
    print(str1, f" completed in {elapsed:.0f} seconds", end='\033[1E')
    # fitnesses = [fitness(g.network) for g in pop.networks]
    # Print information about the best network
    max_fit = max(fitnesses)
    vals.append(max_fit)
    index = fitnesses.index(max_fit)
    gmax = pop.networks[index].network
    if (i % 1 == 0):
        str1 = f"Best Score: {max_fit:8.3g}"
        str2 = "\t   FOV: {:3.0f}\tRange: {:6.3f}\tTurning: {:6.3f}\tSpeed: {:6.3f}\v".format(
            gmax.get_node(0).get("x1"),
            gmax.get_node(0).get("x2"),
            gmax.get_node(0).get("x3"),
            gmax.get_node(0).get("x4"),
        )
        print(str1, str2)
        with open('experiment_eon1.checkpoint.json', 'w') as f:
            f.write(str(pop.as_json(True)))

        # fig, axs = plt.subplots(2, 1)
        # axs[0].hist(fitnesses, bins='auto', range=(min(fitnesses), 0))
        # axs[0].set_title(f"Epoch {i}: Fitness Histogram")
        # axs[0].set_xlabel("Loss")
        # axs[0].set_ylabel("Occurences")
        # axs[1].hist(fitnesses, bins='auto', range=(-500, 0))
        # axs[1].set_xlabel("Loss")
        # axs[1].set_ylabel("Occurences")

        # try:
        #     os.remove('losses.pdf.old')
        # except (FileNotFoundError, OSError):
        #     pass
        # finally:
        #     try:
        #         os.rename('losses.pdf', 'losses.pdf.old')
        #     except (FileNotFoundError, OSError):
        #         pass
        # print("Backed up graph")
        # fig.savefig('losses.pdf')
        # print("saved graph")
        # fig.show()  # Causes ioctl error in WSL
        # print("showed plot")

        # f = open('experiment_eon1.log.txt', 'a')
        # print("Opened log")
        # f.write('\n'.join((time.strftime("%Y%m%d %X"), str1, str2, str(fitnesses), '\n')))
        # print("Wrote to log")
        # f.close()
        # print("closed log")

    # Create the next population based on the fitnesses of the current population
    return evolver.do_epoch(pop, fitnesses, eons_params)
    print("evolved")


try:
    for i in range(500):
        pop = do_epoch(pop)
except KeyboardInterrupt:
    print("Stopping...", flush=True)
    raise

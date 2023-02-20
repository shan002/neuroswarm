import time
import numpy as np
from multiprocessing import Pool

import neuro
import eons

from .utils import *
from .application import *


class Evolver:

    def __init__(self, *, app, eons_params, proc_name, proc_params):

        if not isinstance(eons_params, dict) or not isinstance(proc_params, dict):
            raise TypeError("Passed EONS/Processor parameters must be dictionaries")

        if not isinstance(app, Application):
            raise TypeError("The application must derive from Application")

        self.app = app
        self.eons_params = eons_params
        self.proc_name = proc_name
        self.proc_params = proc_params
        self.sim = proc_name(proc_params)

        if not isinstance(self.sim, neuro.Processor):
            raise TypeError("The processor must derive from neuro.Processor")

        self.eo = eons.EONS(eons_params)
        self.initialize_population()
        self.epoch = 0
        self.fitness_by_epoch = list()

    def initialize_population(self, rng=1, do_print=False):

        if do_print:
            t0 = time.time()

        # Create a template network with the right number of inputs & outputs
        template_net = make_template(self.sim, self.app.n_inputs, self.app.n_outputs)
        self.eo.set_template_network(template_net)

        # Generate a new initial population for this EONS instance
        self.pop = self.eo.generate_population(self.eons_params, rng)

        if do_print:
            print("Initialized population of {} networks in {:8.5f} seconds".format(
                len(self.pop.networks), time.time() - t0))

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass

    def evaluate_population(self, networks):
        return [self.app.fitness(self.sim, network) for network in networks]

    def evaluate_validation(self, network):
        return self.app.validation(self.sim, network)

    def do_epoch(self, do_print=True, add_print='', update_params={}):
        t_start = time.time()

        # Do any pre-epoch operations
        self.pre_epoch()

        # Get the fitness for each network in the population
        t_fs = time.time()
        networks = [nn.network for nn in self.pop.networks]
        self.fitness = self.evaluate_population(networks)
        t_fitness = time.time() - t_fs

        # Track our best network / fitness score
        self.best_network = self.pop.networks[np.argmax(self.fitness)].network
        self.best_fitness = max(self.fitness)
        self.fitness_by_epoch.append(self.best_fitness)

        # Validation score for the best network
        # If an app does not specify a validation score, it should return `None`
        validation = self.evaluate_validation(self.best_network)

        # Evolve the next population with EONS
        t_es = time.time()
        self.pop = self.eo.do_epoch(self.pop, self.fitness, update_params)
        t_eons = time.time() - t_es

        # Do any post-epoch operations
        self.post_epoch()

        t_end = time.time()
        t_total = t_end - t_start

        # Build print status string
        if do_print:
            # Epoch number and fitness score
            ostr = "Epoch {:3d}: {:11.4f}".format(self.epoch, self.best_fitness)

            # Validation score if a validation function is present
            if validation is not None:
                ostr += " {:11.4f}".format(validation)

            # Network size (nodes / edges)
            ostr += " | Neurons: {:3d} Synapses: {:3d}".format(
                self.best_network.num_nodes(), self.best_network.num_edges())

            # Timing information
            ostr += " | Time: {:7.4f} {:7.4f} {:6.4f}".format(t_total, t_fitness, t_eons)

            # Any additional user specified information to print
            if len(add_print) > 0:
                ostr += " --" + add_print

            # Print our fully built string
            print(ostr, flush=True)

        # Increment our epoch counter
        self.epoch += 1

        return self.best_fitness

    def train(self, n_epochs, stop_fitness=None):
        # Start with no "best" score
        best = None

        for epoch in range(n_epochs):
            fit = self.do_epoch()

            # Check if we have a new best
            if best is None or fit > best:
                best = fit
                self.app.save_network(self.best_network)

            # Check for early stopping condition
            if stop_fitness is not None and best > stop_fitness:
                break

    def graph_fitness(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Fitness')
        ax.plot(self.fitness_by_epoch)
        plt.show()


# Helper function for MP Pool mapping
def mp_fitness(bundle):
    app, net, proc_name, proc_params = bundle
    sim = proc_name(proc_params)
    return app.fitness(sim, net)


class MPEvolver(Evolver):

    def __init__(self, *, pool=Pool(), **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def evaluate_population(self, networks):
        bundles = ((self.app, net, self.proc_name, self.proc_params) for net in networks)
        return self.pool.map(mp_fitness, bundles)

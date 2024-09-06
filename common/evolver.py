import os
import time
import numpy as np
from multiprocessing import Pool

import neuro
import eons

from .utils import make_template
from .application import Application

from dataclasses import dataclass

# typing
from typing import Tuple, override


@dataclass
class EpochInfo:
    """Class for the results of an epoch"""
    i: int
    t_start: float
    t_fitness: float
    t_eons: float
    t_end: float
    best_net_id: int
    best_network: neuro.Network
    best_fitness: float
    best_score: float = None
    validation: float | None = None
    fitnesses: Tuple[float, ...] = ()

    @property
    def t_total(self) -> float:
        return self.t_end - self.t_start

    def __str__(self):  # type: ignore[reportImplicitOverride]
        # Epoch number and fitness score
        out = f"Epoch {self.i:3d}: {self.best_fitness:11.4f}"
        out += '' if self.best_score is None else f":{self.best_score:11.4f}"
        # Validation score if a validation function is present
        if self.validation is not None:
            out += " {:11.4f}".format(self.validation)
        # Network size (nodes / edges)
        out += f" | Neurons: {self.best_network.num_nodes():3d} Synapses: {self.best_network.num_edges():3d}"
        # Timing information
        out += f" | Time: {self.t_fitness:7.4f} {self.t_eons:7.4f} {self.t_total:6.4f}"
        return out


class Evolver:

    def __init__(
        self,
        *,
        app,
        eons_params,
        proc_name,
        proc_params,
        stop_fitness=None,
        do_print=True
    ):

        if not isinstance(eons_params, dict) or not isinstance(proc_params, dict):
            raise TypeError("Passed EONS/Processor parameters must be dictionaries")

        if not isinstance(app, Application):
            raise TypeError("The application must derive from Application")

        # if environment variable $EONS_DISABLE_PRINT is set, override `do_print` to be False
        self.do_print = do_print and not os.environ.get("EONS_DISABLE_PRINT", "").strip()

        self.app = app
        self.eons_params = eons_params
        self.proc_name = proc_name
        self.proc_params = proc_params
        self.sim = proc_name(proc_params)

        if not isinstance(self.sim, neuro.Processor):
            raise TypeError("The processor must derive from neuro.Processor")

        # pop our custom eons params so EONS doesn't complain (with defaults)
        self.penalty = eons_params.pop('penalty', None)

        self.stop_fitness = float('inf') if stop_fitness is None else stop_fitness

        self.eo = eons.EONS(eons_params)
        self.initialize_population()
        self.epoch = 0
        self.best: float = float('-inf')
        self.pocket: EpochInfo | None = None
        self.scores_by_epoch = list()

        self.net_callback = lambda net: net

    def initialize_population(self, moa_state=1):
        if self.do_print:
            t0 = time.time()

        # Create a template network with the right number of inputs & outputs
        template_net = make_template(self.sim, self.app.n_inputs, self.app.n_outputs)  # type: ignore[reportAttributeAccessIssue]
        self.eo.set_template_network(template_net)

        # Generate a new initial population for this EONS instance
        self.pop = self.eo.generate_population(self.eons_params, moa_state)

        if self.do_print:
            print("Initialized population of {} networks in {:8.5f} seconds".format(
                len(self.pop.networks), time.time() - t0))

    def pre_epoch(self):
        f = getattr(self.app, 'pre_epoch', None)
        if callable(f):
            f(self)

    def post_epoch(self, epoch_info, new_best):
        f = getattr(self.app, 'post_epoch', None)
        if callable(f):
            f(self, epoch_info, new_best)

    def evaluate_population(self, networks):
        return [self.app.fitness(self.sim, network) for network in self.net_callback(networks)]

    def evaluate_validation(self, network):
        return self.app.validation(self.sim, network)

    def fitness_with_penalty(self, fitnesses, networks):
        # get scores from zipped bundles
        # bundle should be list(zip(networks, fitnesses))
        if self.penalty is None:
            scores = fitnesses
        elif callable(self.penalty):
            scores = self.penalty(fitnesses, networks)
        else:
            # penalize networks by the number of nodes and edges.
            c1, c2 = self.penalty  # this is retrieved from eons_params in __init__
            scores = [
                (fitness - (net.num_nodes() * c1 + net.num_edges() * c2))
                for fitness, net in zip(fitnesses, networks)
            ]  # see 4.5 Multi-Objective Optimization
            # "Evolutionary Optimization for Neuromorphic Systems", Schuman et al. 2020
        return scores

    def do_epoch(self, update_params={}, **kwargs):  # noqa: B006
        t_start = time.time()

        # Do any pre-epoch operations
        self.pre_epoch()

        # Get the fitness for each network in the population
        t_fs = time.time()
        networks = [nn.network for nn in self.pop.networks]
        self.fitness = self.evaluate_population(networks)
        t_fitness = time.time() - t_fs

        # apply penalty function
        scores = self.fitness_with_penalty(self.fitness, networks)

        # bundle
        bundles = enumerate(zip(networks, self.fitness, scores))
        best = max(bundles, key=lambda b: b[-1])  # get best net by score
        topscoring_net_id, topscoring_net, topscoring_fitness, topscore = best
        self.scores_by_epoch.append(topscore)

        # Validation score for the best network
        # If an app does not specify a validation score, it should return `None`
        validation = self.evaluate_validation(topscoring_net)

        # Evolve the next population with EONS
        t_es = time.time()
        self.pop = self.eo.do_epoch(self.pop, scores, update_params)
        t_eons = time.time() - t_es

        # Increment our epoch counter
        self.epoch += 1

        t_end = time.time()

        info = EpochInfo(
            self.epoch,
            t_start,
            t_fitness,
            t_eons,
            t_end,
            topscoring_net_id,
            topscoring_net,
            topscoring_fitness,
            topscore,
            validation,
            tuple(self.fitness),  # every score in the population
        )

        new_best = False
        # save new incumbent solution if it's better
        if topscoring_fitness > self.best:
            new_best = True
            self.best = topscoring_fitness
            self.pocket = info

        if self.do_print:
            print(info)

        # Do any post-epoch operations
        self.post_epoch(info, new_best)

        return info

    def train(self, n_epochs):
        for _epoch in range(n_epochs):
            self.do_epoch()
            if self.stop_fitness is not None and self.best > self.stop_fitness:
                break  # stop early if stopping condition is met
        return self.pocket

    def graph_fitness(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Fitness')
        ax.plot(self.scores_by_epoch)
        plt.show()


# Helper function for MP Pool mapping
def mp_fitness(bundle):
    app, net, proc_name, proc_params = bundle
    sim = proc_name(proc_params)
    return app.fitness(sim, net)


class MPEvolver(Evolver):
    def __init__(self, *, pool=None, **kwargs):
        super().__init__(**kwargs)
        self.pool = Pool() if pool is None else pool

    @override
    def evaluate_population(self, networks):
        bundles = ((self.app, net, self.proc_name, self.proc_params) for net in networks)
        return self.pool.map(mp_fitness, bundles)

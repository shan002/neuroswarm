# from multiprocessing import Pool, TimeoutError
# from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
# import tqdm
import numpy as np
# import neuro
import caspian
# import random
# import argparse
import re
# import os
# import time
# import pathlib
# import matplotlib.pyplot as plt

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.space.transformers import Transformer, Normalize
from skopt.utils import use_named_args

# Provided Python utilities from tennlab framework/examples/common
import common.jsontools as jst
import common.experiment
# from common.experiment import CustomPool
from common.evolver import Evolver
from common.evolver import MPEvolver

import experiment_tenn2 as experiment

# typing
from common.evolver import EpochInfo


class ConnorMillingExperiment(experiment.ConnorMillingExperiment):
    """Tennbots application for TennLab neuro framework & Connor RobotSwarmSimulator (RSS)


    """

    # def __init__(self, args):
    #     super().__init__(args)

    # def fitness(self, processor, network):
    #     return super().fitness(processor, network)


def train(app, args):
    # modified from common.experiment.train(app, args)
    processes = args.processes
    epochs = args.epochs
    max_fitness = args.max_fitness

    if args.population_size is not None:  # if specified, override eons_params file
        app.eons_params["population_size"] = args.population_size
    if args.eons_seed is not None:  # if specified, force EONS seed
        app.eons_params["seed_eo"] = args.eons_seed

    # app.log(f"initialized {args.environment} for training.")

    if processes == 1:
        evolve = Evolver(
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
        )
        # evolve.net_callback = lambda x: tqdm(x,)
    else:
        evolve = MPEvolver(  # multi-process for concurrent simulations
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
            pool=Pool(max_workers=processes),
            # pool=CustomPool(max_workers=processes),
        )
    evolve.print_callback = app.log_status

    try:
        return evolve.train(epochs, max_fitness)
    except KeyboardInterrupt:
        raise


class ScalarTransformer(Transformer):
    """
    Scales each dimension by constant c.

    Parameters
    ----------
    c : [float, int]
        scalar constant

    is_int : bool, default=False
        Round and cast the return value of `inverse_transform` to integer. Set
        to `True` when applying this transform to integers.
    """

    def __init__(self, c: [float, int], is_int: bool = False):
        self.c = c
        self.is_int = is_int

    def transform(self, X):
        X = np.asarray(X)
        X *= self.c

    def inverse_transform(self, X):
        X = np.asarray(X)
        X /= self.c
        if self.is_int:
            X = np.round(X).astype(int)
        return X


class CustomIntSpace(Integer):
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        if transform is None:
            transform = 'identity'
        if isinstance(transform, str):
            super().set_transformer(transform)
        else:
            if not isinstance(transform, Transformer):
                raise ValueError
            self.transformer = transform


def HPO(args):

    args.logfile = None  # disable logging on the app

    dimensions = []

    # EONS Params
    # dimensions.append(Integer(low=1, high=10, name='popsize'))
    dimensions.append(Integer(low=0, high=4, name='EONS_merge_rate'))
    dimensions.append(Integer(low=0, high=4, name='EONS_crossover_rate'))
    dimensions.append(Integer(low=1, high=9, name='EONS_mutation_rate'))
    dimensions.append(CustomIntSpace(low=1, high=3, name='EONS_tournament_size_factor'))
    dimensions.append(CustomIntSpace(low=8, high=10, name='EONS_tournament_best_net_factor'))
    dimensions.append(CustomIntSpace(low=1, high=5, name='EONS_random_factor'))
    dimensions.append(Integer(low=1, high=3, name='EONS_num_mutations'))
    dimensions.append(Integer(low=1, high=9, name='EONS_num_best'))
    dimensions.append(Real(low=0.1, high=1.0, name='node_threshold'))
    dimensions.append(Real(low=0.1, high=1.0, name='edge_weight'))
    dimensions.append(Real(low=0.1, high=1.0, name='edge_delay'))
    # ENCODING Params
    # dimensions.append(Integer(low=3, high=10, name='interval'))
    # dimensions.append(Categorical(categories=['bins', 'flip_flop', 'triangle'], name='fcn'))
    # dimensions.append(Categorical(categories=['2', '4', '8', '10', '12'], name='bins'))
    # dimensions.append(Integer(low=1, high=10, name='charge'))
    # dimensions.append(Categorical(categories=['2', '4', '8', '10', '12'], name='maxspikes'))
    # RUN TIME
    # dimensions.append(Integer(low=11, high=20, name='simtime'))

    # default_parameters = [5, 5, 7, 6, 'flip_flop', '4', 1, '10', 12]

    eons_params = jst.smartload(args.eons_params)

    @use_named_args(dimensions=dimensions)
    def wrapper(**kwargs):
        # inject hyperparameters into app
        #
        # EONS_PREFIX = re.compile('^EONS_')  # '^' is regex string-start anchor
        # for xkey, xval in kwargs.items():
        #     # match eons keys and send them to eons_params
        #     eons_key, matches = EONS_PREFIX.subn('', xkey)
        #     if matches:
        #         eons_params[eons_key] = xval

        eons_params['merge_rate'] = kwargs['EONS_merge_rate'] * 0.1
        eons_params['crossover_rate'] = kwargs['EONS_crossover_rate'] * 0.1
        eons_params['mutation_rate'] = kwargs['EONS_mutation_rate'] * 0.1
        eons_params['tournament_size_factor'] = kwargs['EONS_tournament_size_factor'] * 0.1
        eons_params['tournament_best_net_factor'] = kwargs['EONS_tournament_best_net_factor'] * .1
        eons_params['random_factor'] = kwargs['EONS_random_factor'] * 0.05

        # manually inject these weird ones
        eons_params['node_mutations'].update({'Threshold': kwargs.pop('node_threshold', 1.0)})
        eons_params['edge_mutations'].update({'Weight': kwargs.pop('edge_weight', 0.65)})
        eons_params['edge_mutations'].update({'Delay': kwargs.pop('edge_delay', 0.35)})

        args.eons_params = eons_params

        # run evolutionary training and get best score
        app = ConnorMillingExperiment(args)
        best: EpochInfo = train(app, args)
        return -best.best_score

    result = gp_minimize(
        wrapper,
        dimensions,
        acq_func='gp_hedge',
        n_calls=100,
        n_random_starts=10,
        noise='gaussian',
        random_state=1,
        verbose=True,
    )

    print(result)


def get_parsers(parser, subpar):
    # this is a separate function so we can inherit options from this module
    sp = subpar.parsers

    for sub in sp.values():  # applies to everything
        pass

    for key in ('test', 'run'):  # arguments that apply to test/validation and stdin
        sp[key].add_argument('--network', help="network", default="networks/experiment_tenn2BO.json")

    # Training args
    sp['train'].add_argument('--network', default="networks/experiment_tenn2BO_train.json",
                             help="output network file path.")
    sp['train'].add_argument('--logfile', default=None,
                             help="running log file path.")

    return parser, subpar


def main():
    parser, subpar = common.experiment.get_parsers()
    parser, subpar = experiment.get_parsers(parser, subpar)
    parser, subpar = get_parsers(parser, subpar)

    args = parser.parse_args()

    args.environment = "connorsim_snn_eonsbo-v01"

    # Do the appropriate action
    if args.action == "train":
        HPO(args)
    else:
        print("WARNING: Running from Bayesian Optimization Script, but no training or BO ocurring.")
        print("WARNING: Untested.")
        app = ConnorMillingExperiment(args)
        common.experiment.run(app, args)


if __name__ == "__main__":
    main()

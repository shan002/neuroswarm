from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import common


# from rss.gui import TennlabGUI
import experiment_tenn2
from experiment_tenn2 import ConnorMillingExperiment
import rss.graphing as graphing

# typing:
from typing import override
from swarmsim.world.RectangularWorld import RectangularWorld

from common.argparse import ArgumentError
from common import experiment


def graph_each(world):
    bundles = []
    for i, agent in enumerate(world.population):
        fig, ax = plt.subplots()
        axw = ax.twinx()
        bundles.append(graphing.plot_single(agent, fig, (ax, axw)))
        _h, _l, legend = graphing.label_vwx((fig, (ax, axw)),
                                            title=f"Agent {i} Sensors vs. Control Inputs")
        # make it smaller
        fig.set_figheight(2)
        fig.subplots_adjust(top=0.70, bottom=0.22)
        fig.subplots_adjust(left=0.11, right=0.88)
        legend.set_bbox_to_anchor((0.5, 0.90))
    return bundles


def run(app, args):

    # Set up simulator and network

    proc = None
    net = app.net

    # Run app and print fitness
    world = app.simulate(proc, net)
    fitness = app.extract_fitness(world)
    print(f"Fitness: {fitness:8.4f}")

    import matplotlib.pyplot as plt
    # fig = graphing.plot_multiple(world)
    # fig.suptitle('')
    for fig, _axs in graph_each(world):
        fig.savefig(app.p.ensure_file_parents(f"plots/agent_trajectories_{fig.number}.pdf"))

    fig, _ax = graphing.plot_fitness(world)
    # fig.suptitle('')
    # make it smaller
    fig.set_figheight(2)
    fig.subplots_adjust(bottom=0.22)
    fig.subplots_adjust(left=0.11, right=0.95)
    fig.savefig(app.p.ensure_file_parents(f"plots/fitness.pdf"))

    graphing.export(world, output_file=app.p.ensure_file_parents("agent_trajectories.xlsx"))
    if args.explore:
        app.p.explore()
    plt.show(block=True)
    # TODO: handle when no project

    return fitness


def get_parsers(parser, subpar):
    return parser, subpar


def main():
    parser, subpar = experiment.get_parsers()
    parser, subpar = experiment_tenn2.get_parsers(parser, subpar)
    parser, subpar = get_parsers(parser, subpar)  # modify parser

    args = parser.parse_args()

    args.environment = "connorsim_snn_eons-v01"  # type: ignore[reportAttributeAccessIssue]
    if args.project is None and args.logfile is None:
        args.logfile = "tenn2_train.log"

    app = ConnorMillingExperiment(args)

    args.action = "run"
    run(app, args)


if __name__ == "__main__":
    main()

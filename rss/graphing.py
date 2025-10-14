import pandas as pd
import colorsys
import itertools
import matplotlib.pyplot as plt
import matplotlib
from typing import cast


def hr(h, s, l):  # noqa: E741
    return colorsys.hls_to_rgb(h, l, s)


def plot_fitness(world):
    fig, ax = plt.subplots()
    metric = world.metrics[0]
    ax.plot(metric.value_history)
    if metric.instantaneous:
        ax.set_title(f"Average {metric.name}: {metric.average:0.4f}")
    else:
        ax.set_title(f"Instantaneous {metric.name}: {metric.value:0.4f}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(metric.name)
    return fig, ax


def extract_history(a):
    t = list(range(len(a.history)))  # time
    state, sense, out = list(zip(*a.history))
    v, w = list(zip(*out))
    x, y, theta = list(zip(*state))
    return t, x, y, theta, sense, v, w


def plot_single_artists(a, fig, axs, plot_state=False):
    cr = hr(0.0, 0.9, 0.4)
    cb = hr(0.6, 0.9, 0.4)
    cg = hr(0.3, 0.9, 0.4)
    ax, axw = axs
    ax = cast(plt.Axes, axs[0])
    axw = cast(plt.Axes, axs[1])

    x, _px, _py, _theta, sense, v, w = extract_history(a)

    # create green vertical spanning regions for sensors
    xsen = [1] if sense and sense[0] else []
    xnot = []
    for (xi, si), (xn, sn) in itertools.pairwise(zip(x, sense)):
        if sn > si:
            xsen.append(xn)
        if si > sn:
            xnot.append(xi)
    if sense and sense[-1]:
        xnot.append(len(sense) - 1)

    # breakpoint()
    artists = {}
    ax.cla()
    axw.cla()
    artists['lineplot_sense'] = ax.plot(x, sense, c=cg, label="heck", alpha=0.1)
    # if plot_state:
    #     ax.subplot(111, aspect='equal')
    artists['axvspans_sense'] = [
        ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        for xa, xb in zip(xsen, xnot)
    ]
    artists['lineplot_v'] = ax.plot(x, v, c=cb, label="Velocity v", alpha=0.5)
    artists['lineplot_w'] = axw.plot(x, w, c=cr, label="Turn Rate \\omega", alpha=0.5)

    return fig, (ax, axw), artists


def plot_single(a, fig, axs, plot_state=False):
    fig, axs, _artists = plot_single_artists(a, fig, axs, plot_state)
    return fig, axs


def plot_multiple(world):
    fig = plt.figure()
    ax0 = None
    for i, agent in enumerate(world.population):
        ax = fig.add_subplot(len(world.population), 1, i + 1, sharex=ax0)
        ax0 = ax0 or ax
        axw = ax.twinx()
        plot_single(agent, fig, (ax, axw))
    fig.suptitle(f"Agent Sensors vs. Control Inputs")
    return fig


def export(world, output_file):
    data = [extract_history(agent) for agent in world.population]

    with pd.ExcelWriter(output_file) as writer:
        for i, data in enumerate(data):
            data = list(zip(*data))
            df = pd.DataFrame(data, columns=['t', 'x', 'y', 'angle (rads from east)', 'sense', 'v', 'w'])
            df.to_excel(writer, sheet_name=f'{i}')

import pandas as pd
import colorsys
import itertools


def hr(h, s, l):  # noqa: E741
    return colorsys.hls_to_rgb(h, l, s)


def plot_fitness(world):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    metric = world.metrics[0]
    ax.plot(metric.value_history)
    ax.set_title(f"Instantaneous {metric.name}: {metric.average:0.4f}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(metric.name)
    plt.show()


def extract_history(a):
    t = list(range(len(a.history)))  # time
    state, sense, out = list(zip(*a.history))
    v, w = list(zip(*out))
    x, y, theta = list(zip(*state))
    return t, x, y, theta, sense, v, w


def plot_single(a, fig, ax, plot_state=False):
    cr = hr(0.0, 0.9, 0.4)
    cb = hr(0.6, 0.9, 0.4)
    cg = hr(0.3, 0.9, 0.4)

    x, px, py, theta, sense, v, w = extract_history(a)

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
    ax.cla()
    ax.plot(x, sense, c=cg, label="heck", alpha=0.1)
    # if plot_state:
    #     ax.subplot(111, aspect='equal')
    for xa, xb in zip(xsen, xnot):
        ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
    ax.plot(x, v, c=cb, label="v", alpha=0.5)
    ax.plot(x, w, c=cr, label=r"\omega", alpha=0.5)

    return fig, ax


def plot_multiple(world):
    import matplotlib.pyplot as plt
    fig = plt.gcf()  # get current figure
    for i, agent in enumerate(world.population):
        ax = fig.add_subplot(len(world.population), 1, i + 1)
        plot_single(agent, fig, ax)
    plt.show()


def export(world, output_file):
    data = [extract_history(agent) for agent in world.population]

    with pd.ExcelWriter(output_file) as writer:
        for i, data in enumerate(data):
            data = list(zip(*data))
            df = pd.DataFrame(data, columns=['t', 'x', 'y', 'angle (rads from east)', 'sense', 'v', 'w'])
            df.to_excel(writer, sheet_name=f'{i}')

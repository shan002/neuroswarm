import pandas as pd
import numpy as np
import colorsys
import itertools
from swarmsim.metrics.Circliness import Circliness


def hr(h, s, l):  # noqa: E741
    return colorsys.hls_to_rgb(h, l, s)


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

def plot_multiple_new(world):
    import matplotlib.pyplot as plt

    # Create one extra row: first row for overall circliness
    n_agents = len(world.population)
    fig, axs = plt.subplots(nrows=n_agents + 1, ncols=1, figsize=(8, 12), sharex=True)

    # Plot overall circliness
    if world.metrics and hasattr(world.metrics[0], 'value_history'):
        circliness: Circliness = world.metrics[0]
        fatness = circliness.fatness.value_history
        tangentness = circliness.tangentness.value_history
        plot_overall_circliness(axs[0], circliness.value_history, fatness, tangentness)

    for i, agent in enumerate(world.population):
        plot_single(agent, fig, axs[i + 1])
        axs[i + 1].set_ylabel(f"Agent {i}")
    
    axs[-1].set_xlabel('Timestep')
    plt.tight_layout()
    plt.show()

def plot_overall_circliness(ax, circliness_values, fatness, tangentness):
    import numpy as np
    import matplotlib.ticker as mticker
    import mplcursors

    t = range(len(circliness_values))
    ax.plot(t, circliness_values, color='green', label='Circliness')
    ax.plot(t[len(t)-450:], fatness, color='red', label='Fatness')
    ax.plot(t[len(t)-450:], tangentness, color='blue', label='Tangentness')
    ax.set_title('Circliness over Time')
    ax.set_ylabel('Circliness, Fatness, Tangentness')

    ax.grid(True)
    ax.set_ylim(0, max(1.0, max(circliness_values) * 1.1))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(10))

    # Draw a horizontal line at y = 0.9 (threshold)
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='Threshold 0.9')

    # Determine the first time index where circliness reaches or exceeds 0.9
    x_threshold = None
    for i, v in enumerate(circliness_values):
        if v >= 0.9:
            x_threshold = i
            break
    if x_threshold is not None:
        ax.axvline(x=x_threshold, color='red', linestyle='--', linewidth=1,
                   label=f'Reach 0.9 at t={x_threshold}')

    # Compute and plot the average circliness value
    # avg_value = np.mean(circliness_values)
    # ax.axhline(y=avg_value, color='blue', linestyle='-.', linewidth=1, label=f'Average ({avg_value:.3f})')

    mplcursors.cursor(ax, hover=True)
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))


def build_metrics_df(world):
    metrics_obj = world.metrics[0]
    n = len(metrics_obj.value_history)
    t = list(range(1, n + 1))

    # Full circliness is available for all timesteps.
    circliness = metrics_obj.value_history

    # fatness and tangentness are only plotted for the last 450 timesteps.
    # Padding the initial timesteps with NaN so that the DataFrame has the same length.
    pad_length = max(n - 450, 0)
    # If there are less than 450 timesteps, we don't pad.
    fatness_data = np.array(metrics_obj.fatness.value_history)
    tangentness_data = np.array(metrics_obj.tangentness.value_history)
    if pad_length > 0:
        fatness_full = np.concatenate([np.full(pad_length, np.nan), fatness_data[-450:]])
        tangentness_full = np.concatenate([np.full(pad_length, np.nan), tangentness_data[-450:]])
    else:
        fatness_full = fatness_data
        tangentness_full = tangentness_data

    df_metrics = pd.DataFrame({
        't': t,
        'circliness': circliness,
        'fatness': fatness_full,
        'tangentness': tangentness_full
    })

    return df_metrics



def export(world, output_file):
    data = [extract_history(agent) for agent in world.population]

    with pd.ExcelWriter(output_file) as writer:
        for i, data in enumerate(data):
            data = list(zip(*data))
            df = pd.DataFrame(data, columns=['t', 'x', 'y', 'angle (rads from east)', 'sense', 'v', 'w'])
            df.to_excel(writer, sheet_name=f'{i}')

        df_metrics = build_metrics_df(world)
        df_metrics.to_excel(writer, sheet_name='metrics')

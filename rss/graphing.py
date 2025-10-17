import pandas as pd
import colorsys
import itertools
import matplotlib.pyplot as plt
import matplotlib
from typing import cast


def hr(h, s, l):  # noqa: E741
    return colorsys.hls_to_rgb(h, l, s)


def _suplabels(self, t, info, **kwargs):
    import matplotlib as mpl

    suplab = getattr(self, info['name'], None)

    x = kwargs.pop('x', None)
    y = kwargs.pop('y', None)
    if info['name'] in ['_supxlabel', '_suptitle']:
        autopos = y is None
    elif info['name'] in ['_supylabel', '_supyrlabel']:
        autopos = x is None
    if x is None:
        x = info['x0']
    if y is None:
        y = info['y0']

    if 'horizontalalignment' not in kwargs and 'ha' not in kwargs:
        kwargs['horizontalalignment'] = info['ha']
    if 'verticalalignment' not in kwargs and 'va' not in kwargs:
        kwargs['verticalalignment'] = info['va']
    if 'rotation' not in kwargs:
        kwargs['rotation'] = info['rotation']

    if 'fontproperties' not in kwargs:
        if 'fontsize' not in kwargs and 'size' not in kwargs:
            kwargs['size'] = mpl.rcParams[info['size']]
        if 'fontweight' not in kwargs and 'weight' not in kwargs:
            kwargs['weight'] = mpl.rcParams[info['weight']]

    sup = self.text(x, y, t, **kwargs)
    if suplab is not None:
        suplab.set_text(t)
        suplab.set_position((x, y))
        suplab.update_from(sup)
        sup.remove()
    else:
        suplab = sup
    suplab._autopos = autopos
    setattr(self, info['name'], suplab)
    self.stale = True
    return suplab


def supyrlabel(self, t, **kwargs):
    # docstring from _suplabels...
    info = {'name': '_supyrlabel', 'x0': 0.98, 'y0': 0.5,
            'ha': 'right', 'va': 'center', 'rotation': 'vertical',
            'rotation_mode': 'anchor', 'size': 'figure.labelsize',
            'weight': 'figure.labelweight'}
    return _suplabels(self, t, info, **kwargs)


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
    artists['lineplot_v'] = ax.plot(x, v, c=cb, label="Velocity v", alpha=0.5)
    artists['lineplot_w'] = axw.plot(x, w, c=cr, label="Turn Rate $\\omega$", alpha=0.5)
    artists['lineplot_sense'] = ax.plot(x, sense, c=cg, label="Detection", alpha=0.1)
    artists['axvspans_sense'] = [
        ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        for xa, xb in zip(xsen, xnot)
    ]

    return fig, (ax, axw), artists


def plot_single(a, fig, axs, plot_state=False):
    fig, axs, _artists = plot_single_artists(a, fig, axs, plot_state)
    return fig, axs


def plot_multiple(world):
    fig = plt.figure()
    ax0 = None
    bundles = []
    for i, agent in enumerate(world.population):
        ax = fig.add_subplot(len(world.population), 1, i + 1, sharex=ax0)
        ax0 = ax0 or ax
        axw = ax.twinx()
        bundles.append(plot_single(agent, fig, (ax, axw)))
    label_vwx(*bundles, title=f"Agent Sensors vs. Control Inputs")
    return fig


def label_vwx(*plots, title=""):
    for fig, (ax, axw) in reversed(plots):
        # grab fig, axes from the last plot
        try:
            # grab the artists and labels from the primary axis
            handles, labels = ax.get_legend_handles_labels()
            a, b = axw.get_legend_handles_labels()
            # add the artists and labels from the twinx (red plot, right side)
            handles.insert(-1, *a)
            labels.insert(-1, *b)
        except Exception:
            continue
        fig = fig
        break

    # show the legend
    # if len(plots) == 1:
    #     legend = axw.legend(handles=handles,
    #                         loc='lower center', ncol=3, fancybox=True, shadow=True)
    #     ax.set_xlabel("Time since start (seconds)", loc='center')
    #     ax.set_ylabel("Forward Velocity (m/s)")
    #     if title:
    #         ax.set_title(title)
    #     axw.yaxis.label_position = 'right'
    #     axw.yaxis.labelpad = 10.0
    #     axw.set_ylabel("Angular Velocity (rad/s)")
    #     fig.subplots_adjust(right=(1 - fig.subplotpars.left))
    # else:
    if True:
        bbox = {'bbox_to_anchor': (0.5, 0.95)} if title else {}  # only shift the legend downwards if title shown
        legend = fig.legend(handles=handles,
                            loc='upper center', ncol=3, fancybox=True, shadow=True, **bbox)
        if title:
            fig.suptitle(title)
        top = 0.87 if title else 0.88
        fig.supxlabel("Time since start (seconds)", ha='center')
        fig.supylabel("Forward Velocity (m/s)")
        supyrlabel(fig, "Angular Velocity (rad/s)")
        fig.subplots_adjust(top=top, hspace=0.08, right=(1 - fig.subplotpars.left), bottom=0.1)

    # set the linewidth of each legend object
    for obj in legend.legend_handles:
        obj.set_linewidth(3.0)  # pyright: ignore[reportAttributeAccessIssue]
    s = legend.legend_handles[-1]  # the last column should be the vertical lines.
    s.set_linewidth(10.0)  # Draw that thicker in the legend   # pyright: ignore[reportAttributeAccessIssue]
    s.set_alpha(0.25)
    return handles, labels, legend


def export(world, output_file):
    data = [extract_history(agent) for agent in world.population]

    with pd.ExcelWriter(output_file) as writer:
        for i, data in enumerate(data):
            data = list(zip(*data))
            df = pd.DataFrame(data, columns=['t', 'x', 'y', 'angle (rads from east)', 'sense', 'v', 'w'])
            df.to_excel(writer, sheet_name=f'{i}')

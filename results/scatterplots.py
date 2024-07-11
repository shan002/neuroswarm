import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tomllib
import re

import colorsys


def load_data(data_path):
    df = pd.read_csv(data_path)
    df["turning_radius_0"] = [min(abs(i / j), 250) for i, j in zip(df["forward_rate_0"], df["turning_rate_0"])]
    df["turning_radius_1"] = [min(abs(i / j), 250) for i, j in zip(df["forward_rate_1"], df["turning_rate_1"])]
    return df


def plot_evolution(data):
    x = data["time"]
    min_time = min(x)
    x = [elem - min_time for elem in x]
    y = data["Circliness"]
    plt.scatter(x, y, s=5)
    plt.title("Population Circliness during Evolutionary Optimization")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Circliness (Eq. 9)")
    plt.show()


def load_tsv_file(path):
    import csv
    out = []
    with open(path, 'r') as f:
        f.readline()
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            out.append(row)
    return out


def load_txt(
    path: str,
    start: int = 0,
    end: int | None = None
) -> list[str]:
    with open(path, 'r') as f:
        for _ in range(start):
            next(f)  # skip `start` lines
        return f.readlines(end)


def extract_tenn2_data(lines: list[str]):
    RE_EPOCH = re.compile(r'[1-2]\d{7}\s?[\d:]{0,8}\s?>>>\s?Epoch\s*(?P<epoch>\d+):\s*(?P<score>[\d.]+):\s*(?P<fit>[\d.]+)\s\|\sNeurons:\s*(?P<neur>\d+)\sSynapses:\s*(?P<syn>\d+)\s\|\sTime:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)')  # noqa: E501
    matches = [RE_EPOCH.search(line) for line in lines]
    data = [match.groups() for match in matches if match]
    # {'epoch', 'score', 'fit', 'neur', 'syn', 6, 7, 8}
    return data


def scatter_time(fig, ax, data, label, colors, shape, epoch_vline=None, annotate_last=False):
    d, l, c, s = data, label, colors, shape  # noqa
    time_start = float(d[1][0])
    x = []
    y = []
    for i in range(1, len(d)):
        fitnesses = eval(d[i][2])
        t = float(d[i][0]) - time_start
        if i == epoch_vline:
            ax.axvline(t, c=c, alpha=0.2)
            print(t)
        for f in fitnesses:
            x.append(t)
            y.append(f)

    fitnesses = eval(d[-1][2])
    t = float(d[i][0]) - time_start
    best = max(fitnesses)
    plt.annotate(f'({int(t)}, {best:0.4})', (t, best), textcoords="offset points", xytext=(0, 3), ha='right', color=c)

    ax.scatter(x, y, c=c, s=2, label=l, alpha=0.8)


def plot_epochs(fig, ax, data, label, colors, shape, epochs=None):
    print(len(data))
    x = []
    y = []
    for i in range(1, len(data)):
        fitnesses = eval(data[i][2])
        best_fitness = max(fitnesses)
        x.append(i)
        y.append(best_fitness)
    ax.plot(x, y, c=colors, label=label, alpha=0.8)


def scatter_epochs(fig, ax, data, label, colors, shape, epochs=None):
    print(len(data))
    x = []
    y = []
    for i in range(1, len(data)):
        fitnesses = eval(data[i][2])
        for f in fitnesses:
            x.append(i)
            y.append(f)
    ax.scatter(x, y, c=colors, s=2, label=label, alpha=0.2)


def plot_compare2():
    kevins = (load_tsv_file("s/20240422_1000t_n10_p100_e1000_s23.tsv"), "SNN", "red", "*")
    connor = (load_tsv_file("20240425-connormill_10n_1000t_p100_e1000_(0)_raw.tsv"), "Symbolic", "blue", "^")

    fig, ax = plt.subplots()

    # for d, l, c, s in [kevin, conno]:
    #     scatter_time(fig, ax, d, l, c, s)
    #     # plot_epochs(fig, ax, d, l, c, s)
    scatter_epochs(fig, ax, *kevins)
    # scatter_time(fig, ax, *kevins, epoch_vline=100)
    # scatter_time(fig, ax, *conno)
    # scatter_time(fig, ax, *connor, epoch_vline=100)
    scatter_epochs(fig, ax, *connor)
    legend = ax.legend(loc='lower right', markerscale=5)
    for handle in legend.legend_handles:
        handle.set_alpha(0.5)
    # plt.xlabel("Time (seconds)")
    plt.xlabel("Epochs")
    plt.ylabel("Fitness")
    plt.title("Population Fitness Distribution during G.A.")

    plt.subplots_adjust(left=0.11, right=0.95)
    plt.show()


def plot_netsize():
    def hr(h, s, l):  # noqa: E741
        return colorsys.hls_to_rgb(h, l, s)

    lines = load_txt('20240422_tenn2_train.log')
    data = extract_tenn2_data(lines)

    epochs, score, fit, neurons_, synapses, *_ = zip(*data)

    epochs, neurons_, synapses = [[float(x) for x in vec] for vec in (epochs, neurons_, synapses)]

    fig, ax = plt.subplots()

    c_nrn = hr(0.35, 0.6, 0.45)
    c_syn = hr(0.7, 0.8, 0.4)

    ax.plot(epochs, neurons_, c=c_nrn, label='# of Neurons', alpha=0.9)
    ax.plot(epochs, synapses, c=c_syn, label='# of Synapses', alpha=0.9)

    plt.annotate(f'{int(neurons_[-1])} ', (epochs[-1], neurons_[-1]), textcoords="offset points", xytext=(2, -8), ha='left', color=c_nrn)
    plt.annotate(f'{int(synapses[-1])} ', (epochs[-1], synapses[-1]), textcoords="offset points", xytext=(2, -8), ha='left', color=c_syn)

    ax.legend(loc='upper right')
    # plt.xlabel("Time (seconds)")
    plt.xlabel("Epochs")
    plt.ylabel(" ")
    plt.title("SNN Network Size")

    fig.set_figheight(2)
    plt.subplots_adjust(bottom=0.24)

    plt.subplots_adjust(left=0.11, right=0.95)
    plt.show()


def plot_shadowplot():
    def hr(h, s, l):  # noqa: E741
        return colorsys.hls_to_rgb(h, l, s)

    c_conno = hr(0.6, 0.9, 0.4)
    c_kevin = hr(0.0, 0.9, 0.4)

    c0 = (load_tsv_file("20240425-connormill_10n_1000t_p100_e1000_(0)_raw.tsv"), "Symbolic", c_conno, "^")  # only
    k2 = (load_tsv_file("s/20240422_1000t_n10_p100_e1000_s23.tsv"), "SNN", c_kevin, "*")  # best

    fpaths = [
        "s/20240417_1000t_n10_p100_e1000_s20.tsv",
        "s/20240418_1000t_n10_p100_e1000_s21.tsv",
        "s/20240420_1000t_n10_p100_e1000_s22.tsv",
        "s/20240422_1000t_n10_p100_e1000_s23.tsv",
        "s/20240423_1000t_n10_p100_e1000_s24.tsv",
        "s/20240424_1000t_n10_p100_e1000_s25.tsv",
    ]

    kdata = [load_tsv_file(f) for f in fpaths]
    fitnesses_per_run = [[max(eval(fitnesses), default=0) for time, nth_epoch, fitnesses in run] for run in kdata]
    fitnesses_per_epoch = list(zip(*fitnesses_per_run))

    maxbest_per_epoch = [max(fitnesses) for fitnesses in fitnesses_per_epoch]
    minbest_per_epoch = [min(fitnesses) for fitnesses in fitnesses_per_epoch]

    epoch_idxs = range(len(maxbest_per_epoch))

    fig, ax = plt.subplots()

    ax.plot(epoch_idxs, maxbest_per_epoch, c=c_kevin, alpha=0.4)
    ax.plot(epoch_idxs, minbest_per_epoch, c=c_kevin, alpha=0.4)

    ax.fill_between(epoch_idxs, maxbest_per_epoch, minbest_per_epoch, facecolor=c_kevin, alpha=0.12)

    plot_epochs(fig, ax, *c0)
    plot_epochs(fig, ax, *k2)

    # snn label best coordinates
    best = max([fit for run in fitnesses_per_run for fit in run])
    firstbest_epoch = min([run.index(best) for run in fitnesses_per_run if best in run])
    plt.annotate(f'({firstbest_epoch}, {best:0.5})', (firstbest_epoch, best), textcoords="offset points", xytext=(0, 3), ha='right', color=c_kevin)

    # connor label best coordinates
    bests = [max(eval(fitnesses), default=0) for time, nth_epoch, fitnesses in c0[0]]
    best = max(bests)
    firstbest_epoch = bests.index(best)
    plt.annotate(f'({firstbest_epoch}, {best:0.5})', (firstbest_epoch, best), textcoords="offset points", xytext=(0, 3), ha='left', color=c_conno)

    ax.legend(loc='lower right')
    # plt.xlabel("Time (seconds)")
    plt.xlabel("Epochs")
    # plt.ylabel("λ (Circliness)")
    plt.ylabel("Fitness")
    plt.title("Population Best Fitness during G.A.")

    # fig.set_figheight(2)
    # plt.subplots_adjust(top=0.95, bottom=0.24)
    # plt.axis([-20, 450, 0.92, 1.01])

    plt.subplots_adjust(left=0.11, right=0.95)
    plt.show()


def plot_compare_multi():
    def hr(h, s, l):  # noqa
        return colorsys.hls_to_rgb(h, l, s)

    k1 = (load_tsv_file("nono-1000t_n10_p25_s20_e100.tsv"), "SNN, P=25", hr(0.0, 0.9, 0.7), "*")
    k2 = (load_tsv_file("nono-1000t_n10_p100_s20_e100truncated.tsv"), "SNN, P=100", hr(0.0, 0.9, 0.4), "*")
    c1 = (load_tsv_file("connormill_10n_1000t_p25_s1_e100-2_raw.tsv"), "Sym, P=25", hr(0.6, 0.9, 0.7), "^")
    c2 = (load_tsv_file("connormill_10n_1000t_p100_e100_s1_raw.tsv"), "Sym, P=100", hr(0.6, 0.9, 0.4), "^")

    fig, ax = plt.subplots()

    plot_epochs(fig, ax, *c1)
    plot_epochs(fig, ax, *c2)
    plot_epochs(fig, ax, *k1)
    plot_epochs(fig, ax, *k2)
    ax.legend(loc='lower right')
    # plt.xlabel("Time (seconds)")
    plt.xlabel("Epochs")
    # plt.ylabel("λ (Circliness)")
    plt.ylabel("Fitness")
    plt.title("Population Best Fitness during G.A.")

    plt.subplots_adjust(left=0.11, right=0.95)
    plt.show()


def plot_evolution_over_generations(data):
    x = data["time"]
    min_time = min(x)

    x = range(1, 101)
    y = [np.mean(data.loc[data["gen"] == (i - 1)]["Circliness"]) for i in x]

    plt.plot(x, y)
    plt.title("Average Population Circliness during Evolutionary Optimization")
    plt.xlabel("Generation")
    plt.ylabel("Average Population Circliness (Eq. 9)")
    plt.show()


def plot_heatmap(data):
    ax = sns.relplot(
        data=data,
        x="turning_radius_0", y="turning_radius_1", hue="Circliness", size="Circliness",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )
    plt.show()


def plot_initialization_boxplot():
    def hr(h, s, l):  # noqa: E741
        return colorsys.hls_to_rgb(h, l, s)

    def import_test_data(fpath):
        with open(fpath, 'rb') as f:
            tom = tomllib.load(f)
        if 'seedrange' in tom:
            seed = list(range(*tom['seedrange']))
        else:
            seed = None
        fitnesses = tom['fitnesses']
        return seed, fitnesses

    c_conno = hr(0.6, 0.9, 0.4)
    c_kevin = hr(0.0, 0.9, 0.4)

    _, conno = import_test_data('20240502-connormill-test_data.toml')
    _, kevin = import_test_data('20240502-tenn2_mill-test_data.toml')

    conno_best = 0.988195224312768
    kevin_best = 0.9934842372508302

    # for k, c in zip(kevin, conno):
    #     print(f"c: {c:.5f},\tk: {k:.5f}")

    fig, ax = plt.subplots()

    violins = ax.violinplot(
        (conno, kevin),
        positions=[0, 1],
        showmeans=True,
        # showmedians=True,
        showextrema=False,
        vert=False,
        widths=[0.6, 0.6],
        )

    ax.set_yticks([0, 1],
                  labels=["Symbolic", "SNN"])
    violins['bodies'][0].set_facecolor(c_conno)
    violins['bodies'][0].set_edgecolor(c_conno)
    violins['bodies'][1].set_facecolor(c_kevin)
    violins['bodies'][1].set_edgecolor(c_kevin)
    violins['cmeans'].set_color('black')
    violins['cmeans'].set_alpha(0.8)
    # breakpoint()

    # plt.annotate(f'{int(synapses[-1])} ', (epochs[-1], synapses[-1]), textcoords="offset points", xytext=(2, -8), ha='left', color=c_syn)

    ax.scatter([conno_best], [0], marker='o', color=c_conno, s=30, zorder=3, alpha=0.5)
    ax.scatter([kevin_best], [1], marker='o', color=c_kevin, s=30, zorder=3, alpha=0.5)

    # ax.legend(loc='upper right')
    # plt.xlabel("Time (seconds)")
    plt.xlabel("λ (Circliness)")
    # plt.ylabel(" ")
    plt.title("Circliness Distribution for Random Spawn")

    fig.set_figheight(2)
    plt.subplots_adjust(top=0.82, bottom=0.24)

    plt.subplots_adjust(left=0.14, right=0.95)
    plt.show()


matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
# matplotlib.rcParams['legend.fontsize'] = 10
# matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300

if __name__ == "__main__":
    # data = load_data("../out/comp_w_kevin/CMAES/genomes.csv")
    # aggregate_data(data, "P50_T1000_N10.tsv")
    plot_compare2()
    # plot_compare_multi()
    plot_shadowplot()
    # plot_netsize()
    # plot_initialization_boxplot()

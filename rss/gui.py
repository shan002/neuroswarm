import itertools as it

import pygame
import numpy as np
from math import pi as PI

from swarmsim.gui.agentGUI import DifferentialDriveGUI

from rss.graphing import plot_single_artists, extract_history, hr

# typing
from typing import override
from swarmsim.agent.MazeAgent import MazeAgent

matplotlib = None


def forward_axvspans(x, sense, offset=0):
    # create green vertical spanning regions for sensors
    #     [////]    [///]
    # xsen^    ^xnot
    xsen = [x[0]] if sense and sense[0] else []
    xnot = []
    for (xi, si), (xn, sn) in it.pairwise(zip(x, sense)):
        if sn > si:
            xsen.append(xn + offset)
        if si > sn:
            xnot.append(xi + offset)
    if sense and sense[-1]:
        xnot.append(x[-1])

    return xsen, xnot


class TennlabGUI(DifferentialDriveGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fig = None
        self.axs = []
        self.artists = {}
        self.last_drawn = -1

    # def set_selected(self, agent: MazeAgentCaspian):
    #     super().set_selected(agent)

    @override
    def draw(self, screen, zoom=1.0):
        super().draw(screen, zoom)
        if pygame.font:
            if self.selected:
                a: MazeAgent = self.selected[0]
                if a.controller.neuron_counts is not None:
                    self.appendTextToGUI(screen, f"outs: {a.controller.neuron_counts}")

                if self.time > self.last_drawn:
                    self.plot_single(a)

                # if not was_plotted and self.has_plotted:
                #     self.update_legend()

                self.plt_update_show()

    def plt_update_show(self):
        if self.fig:
            self.fig.canvas.draw_idle()  # draw when events are processed
            self.fig.canvas.flush_events()  # process events
            # plt.pause(0.0001)

    def setup_graph_single(self, agent):
        if not self.check_matplotlib():
            return
        self.setup_figure()
        if not getattr(agent, 'history', False):
            return False

        cr = hr(0.0, 0.9, 0.4)
        cb = hr(0.6, 0.9, 0.4)
        cg = hr(0.3, 0.9, 0.4)
        ax, axw = self.axs

        x, _px, _py, _theta, sense, v, w = extract_history(agent)

        xsen, xnot = forward_axvspans(x, sense)

        # breakpoint()
        ar = self.artists
        ax.cla()
        axw.cla()
        ar['lineplot_v'] = ax.plot(x, v, c=cb, label="Velocity v", alpha=0.5)
        ar['lineplot_w'] = axw.plot(x, w, c=cr, label="Turn Rate $\\omega$", alpha=0.5)
        ar['lineplot_sense'] = ax.plot(x, sense, c=cg, label="heck", alpha=0.1)
        # if plot_state:
        #     ax.subplot(111, aspect='equal')
        ar['axvspans_sense'] = [
            self.new_axvspan(ax, xa, xb)
            for xa, xb in zip(xsen, xnot)
        ]

        # for li in self.artists.values():
        #     for a in li:
        #         a.set_animated(True)

        self.update_legend()
        self.plt_update_show()

    def on_set_selected_single(self, agent):
        super().on_set_selected_single(agent)
        self.setup_graph_single(agent)

    def on_selected_event(self, prev, new):
        super().on_selected_event(prev, new)

    def plot_single(self, agent):
        ar = self.artists
        if not getattr(agent, 'history', False):
            return False
        firstplot = not self.artists
        if firstplot:
            self.setup_graph_single(agent)
        x, _px, _py, _theta, sense, v, w = extract_history(agent)
        ar['lineplot_v'][0].set_data(x, v)
        ar['lineplot_w'][0].set_data(x, w)
        ar['lineplot_sense'][0].set_data(x, sense)
        self.update_axvspans(self.axs[0], x, sense)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        # for li in self.artists.values():
        #     for a in li:
        #         a.axes.draw_artist(a)
        # if firstplot:
        # self.update_legend()
        self.last_drawn = self.time

    def new_axvspan(self, ax, xa, xb):
        return ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')

    def update_axvspans(self, ax, x, sense):
        """Update the green vertical spans for the binary sensor state."""
        try:
            last = self.artists['axvspans_sense'][-1]  # the last green span
            x0 = last.get_x()  # the beginning of the last green span
        except IndexError:  # no green spans
            last = None
            x0 = 0  # check the whole history if no spans
        # Build the start/end pairs of green spans since the last update
        # including the last span that was updated.
        xsen, xnot = forward_axvspans(x[x0:], sense[x0:])
        # old: [////]    [///]
        # new: [////]    [///|/]   (/////) (///) <- new spans to be made
        #              x0^   |-> new
        #                |-->**| <- expand width of last
        for xs, xn in zip(xsen, xnot):
            if xs == x0 and last:
                last.set_width(xn - xs)  # expand the last green span
            else:  # make new green spans since last green span update
                self.artists['axvspans_sense'].append(
                    self.new_axvspan(ax, xs, xn)
                )

    @staticmethod
    def check_matplotlib():
        global matplotlib, plt
        if matplotlib is None:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except ImportError:
                matplotlib = False
            plt.ion()
        return matplotlib

    def setup_figure(self):
        if not self.fig:
            self.fig, ax = plt.subplots()
            axw = ax.twinx()
            self.axs: list[plt.Axes] = [ax, axw]
            self.artists = {}
            self.fig.show()
        return self.fig, self.axs

    def update_legend(self):
        # grab the artists and labels from the primary axis
        handles, labels = self.axs[0].get_legend_handles_labels()
        # add the artists and labels from the twinx (red plot, right side)
        a, b = self.axs[1].get_legend_handles_labels()
        handles.insert(-1, *a)
        labels.insert(-1, *b)

        self.axs[0].set_xlabel("Time since start (seconds)", loc='center')
        self.axs[0].set_ylabel("Forward Velocity (m/s)")
        self.axs[1].yaxis.label_position = 'right'
        self.axs[1].yaxis.labelpad = 10.0
        self.axs[1].set_ylabel("Angular Velocity (rad/s)")
        self.axs[0].set_title(f"Agent {self.selected[0].name} Sensor State and Speeds")
        self.fig.subplots_adjust(right=(1 - self.fig.subplotpars.left))

        # show the legend
        legend = self.axs[1].legend(handles=handles, loc='lower center', ncol=3, fancybox=True, shadow=True)

        # set the linewidth of each legend object
        for obj in legend.legend_handles:
            obj.set_linewidth(3.0)  # pyright: ignore[reportAttributeAccessIssue]
        s = legend.legend_handles[-1]  # the last column should be the vertical lines.
        s.set_linewidth(10.0)  # Draw that thicker in the legend   # pyright: ignore[reportAttributeAccessIssue]
        s.set_alpha(0.25)

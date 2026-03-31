import pygame
import numpy as np
from math import pi as PI

from swarmsim.gui.agentGUI import DifferentialDriveGUI

from rss.graphing import plot_single

# typing
from typing import override
from swarmsim.agent.MazeAgent import MazeAgent

matplotlib = None


class TennlabGUI(DifferentialDriveGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fig, self.ax = None, None

    def appendTextToGUI(self, screen, text, *args, **kwargs):
        if isinstance(text, str) and text.startswith("sees:"):
            return
        return super().appendTextToGUI(screen, text, *args, **kwargs)

    # def set_selected(self, agent: MazeAgentCaspian):
    #     super().set_selected(agent)

    @override
    def draw(self, screen, zoom=1.0):
        super().draw(screen, zoom)
        if pygame.font:
            if self.selected:
                a: MazeAgent | None = self.selected
                if isinstance(a, list):
                    a = a[0] if a else None
                if a is None or not hasattr(a, "controller"):
                    return
                if getattr(a, "sensors", None):
                    sense_val = a.sensors[0].current_state
                    self.appendTextToGUI(screen, f"sense: {sense_val}")
                v = w = None
                if getattr(a, "history", None):
                    _state, _sense, out = a.history[-1]
                    v, w = out
                elif getattr(a.controller, "requested", None):
                    v, w = a.controller.requested
                if v is not None and w is not None:
                    self.appendTextToGUI(screen, f"v: {v:.3f}, w: {w:.3f}")
                if a.controller.neuron_counts is not None:
                    self.appendTextToGUI(screen, f"outs: {a.controller.neuron_counts}")
                self.graph_selected()

    @staticmethod
    def check_matplotlib():
        global matplotlib, plt
        if matplotlib is None:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                plt.ion()
            except ImportError:
                matplotlib = False
        return matplotlib

    def graph_selected(self):
        self.graph_single(self.selected)

    def graph_single(self, a):

        if not self.check_matplotlib():
            return
        if not getattr(a, "history", False):
            return False
        if not self.fig:
            self.fig, self.ax = plt.subplots()

        if not self.sim_paused:
            plot_single(a, self.fig, self.ax)

        plt.draw()
        plt.show()
        plt.pause(0.001)

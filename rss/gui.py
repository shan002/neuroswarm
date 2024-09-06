import pygame
import numpy as np
from math import pi as PI

from novel_swarms.gui.agentGUI import DifferentialDriveGUI

from rss.graphing import plot_single

# typing
from typing import override
from novel_swarms.agent.MazeAgent import MazeAgent

matplotlib = None


class TennlabGUI(DifferentialDriveGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fig, self.ax = None, None

    # def set_selected(self, agent: MazeAgentCaspian):
    #     super().set_selected(agent)

    @override
    def draw(self, screen, zoom=1.0):
        super().draw(screen, zoom)
        if pygame.font:
            if self.selected:
                a: MazeAgent = self.selected
                if a.neuron_counts is not None:
                    self.appendTextToGUI(screen, f"outs: {a.neuron_counts}")
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
        if not getattr(a, 'history', False):
            return False
        if not self.fig:
            self.fig, self.ax = plt.subplots()

        plot_single(a, self.fig, self.ax)

        plt.draw()
        plt.show()
        plt.pause(0.001)

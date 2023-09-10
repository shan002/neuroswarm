import numpy as np
import math
from .Metric import Metric


class _MetricHelper(Metric):
    def center_of_mass(self):
        positions = [(a.x, a.y) for a in self.population]
        center: [float, float] = np.average(positions, axis=0)
        return center


class Fatness2(_MetricHelper):
    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a - b)

    def _calculate(self):
        # calculate average position of all agents
        mu = self.center_of_mass()

        # calculate distance of each agent to mu, save the largest and smallest
        distances = [self.distance((agent.x, agent.y), mu) for agent in self.population]
        rmin = min(distances)
        rmax = max(distances)

        # calculate Fatness but opposite (0 is fat, 1 is perfect circle formation)
        return (rmin ** 2) / (rmax ** 2)


class Fatness(Fatness2):
    def _calculate(self):
        # calculate Fatness (eq(6) from C. Taylor, The impact of catastrophic collisions..., 2021)
        return 1 - super()._calculate()


class Tangentness(_MetricHelper):
    @staticmethod
    def tangentness_inner(agent, mu):
        # inner part of tangentness sum, inside the |abs|
        '''
            p = agent.getPosition()
            d = p - mu
            dnorm = np.linalg.norm(d)
            d = np.zeros(2) if dnorm == 0 else d / np.linalg.norm(d)

            v = agent.getVelocity()
            vnorm = np.linalg.norm(v)
            v = np.zeros(2) if vnorm == 0 else v / vnorm

            return abs(np.dot(d, v))
        '''
        p = (agent.x, agent.y)
        theta = agent.yaw
        d = p - mu
        # dnorm = np.linalg.norm(d)
        # d = np.zeros(2) if dnorm == 0 else d / np.linalg.norm(d)
        d_x, d_y = d
        beta = math.atan2(d_y, d_x)
        alpha = theta - beta
        return abs(math.cos(alpha))

    def _calculate(self):
        # calculate average position of all agents
        mu = self.center_of_mass()
        n = len(self.population)

        # calculate Tangentness
        return np.sum([self.tangentness_inner(agent, mu) for agent in self.population]) / n


class Circliness(_MetricHelper):
    def __init__(self, history_size=100, avg_history_max=100, regularize=False, **kwargs):
        if regularize:
            raise NotImplementedError
        self.tangentness = Tangentness(history_size=avg_history_max,)
        self.fatness = Fatness(history_size=avg_history_max,)
        super().__init__(history_size=history_size, **kwargs)

    def _calculate(self):
        _, tau_ = self.tangentness.out_average()
        _, phi_ = self.fatness.out_average()

        return 1 - max(phi_, tau_)

    def calculate(self):
        self.tangentness.calculate()
        self.fatness.calculate()

        self.set_value(self._calculate())

    @property
    def population(self):
        return self.tangentness.population

    @population.setter
    def population(self, x):
        self.tangentness.population = x
        self.fatness.population = x

    @property
    def world_radius(self):
        return self.tangentness.world_radius

    @world_radius.setter
    def world_radius(self, x):
        self.tangentness.world_radius = x
        self.fatness.world_radius = x

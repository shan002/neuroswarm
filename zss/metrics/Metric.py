from typing import Tuple

from numpy import average


class Metric():
    def __init__(self, name: str = None, history_size=100, num_processes=1):
        self.current_value = 0
        self.history_size = history_size
        self.value_history = []
        self.name = self.__class__.__name__ if name is None else name
        self.population = []
        self.world_radius = None
        self.num_processes = num_processes

    def _calculate(self):
        pass

    def calculate(self):
        self.set_value(self._calculate())

    def set_value(self, value):
        # Keep Track of the [self.history_size] most recent values
        self.value_history.append(value)
        if self.history_size is not None and len(self.value_history) > self.history_size:
            self.value_history = self.value_history[1:]

        self.current_value = value

    def out_current(self) -> Tuple:
        return (self.name, self.value_history[-1])

    def out_average(self) -> Tuple:
        return (self.name, average(self.value_history))

    def reset(self):
        self.current_value = 0
        self.value_history = []

    def draw(self, screen):
        pass

    def as_config_dict(self):
        return {"name": self.name, "history_size": self.history_size}

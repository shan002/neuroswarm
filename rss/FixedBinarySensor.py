from __future__ import annotations

from typing import Iterable

from swarmsim.sensors.AbstractSensor import AbstractSensor


class FixedBinarySensor(AbstractSensor):
    """Return a predefined or random binary sense value each timestep."""

    config_vars = AbstractSensor.config_vars + [
        "mode",
        "sequence",
        "repeat",
        "random_p",
        "time_step_between_sensing",
    ]

    def __init__(
        self,
        agent=None,
        parent=None,
        mode: str = "sequence",
        sequence: Iterable[int] | None = None,
        repeat: bool = True,
        random_p: float = 0.5,
        time_step_between_sensing: int = 1,
        seed=None,
        **kwargs,
    ) -> None:
        super().__init__(agent=agent, parent=parent, seed=seed, **kwargs)
        self.mode = mode
        self.sequence = list(sequence) if sequence is not None else [0]
        self.repeat = repeat
        self.random_p = random_p
        self.time_step_between_sensing = max(1, int(time_step_between_sensing))
        self._step_count = 0
        self._seq_idx = 0

    def _next_sequence_value(self) -> int:
        if not self.sequence:
            return 0
        if self._seq_idx < len(self.sequence):
            value = self.sequence[self._seq_idx]
        else:
            if not self.repeat:
                value = self.sequence[-1]
            else:
                self._seq_idx = 0
                value = self.sequence[self._seq_idx]
        self._seq_idx += 1
        return int(bool(value))

    def step(self, world):
        if self.agent is None and self.static_position is None:
            raise Exception(
                "Either a parent Agent or a static position must be provided to the sensor."
            )

        self._step_count += 1
        if self._step_count % self.time_step_between_sensing != 0:
            return

        if self.mode == "random":
            self.current_state = int(self.rng.random() < self.random_p)
        else:
            self.current_state = self._next_sequence_value()

        # Used by controller to color agents.
        self.detection_id = int(bool(self.current_state))

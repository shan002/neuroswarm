import random
from functools import cached_property

import numpy as np
from novel_swarms.agent.control.AbstractController import AbstractController


class SmartRunnerController(AbstractController):
    """
    When no hunter is in sight → head toward the goal.
    When a hunter is spotted → pick one of:
      • freeze
      • dodge-left
      • dodge-right
    and stick with that action until either:
      1) the hunter disappears, or
      2) the hunter has been in sight for `sight_threshold` timesteps,
         in which case we pick a new random maneuver.
    """

    def __init__(
        self,
        agent=None,
        parent=None,
        target_name="goal",
        speed=0.1,
        turn_rate=5.0,
        # evasion probabilities (must sum to 1)
        freeze_prob=0.2,
        dodge_left_prob=0.4,
        dodge_right_prob=0.4,
        # how many consecutive timesteps of “hunter in sight” triggers a new pick
        sight_threshold=200,
    ):
        super().__init__(agent=agent, parent=parent)

        # goal-seeking params
        self.target_name = (
            target_name.lower() if isinstance(target_name, str) else target_name
        )
        self.speed = speed
        self.turn_rate = turn_rate

        # evasion probabilities
        assert abs(freeze_prob + dodge_left_prob + dodge_right_prob - 1.0) < 1e-6
        self.freeze_prob = freeze_prob
        self.dodge_left_prob = dodge_left_prob
        self.dodge_right_prob = dodge_right_prob

        # sight-based re-pick params
        self.sight_threshold = sight_threshold
        self._sight_counter = 0

        # current evasion action: None | "freeze" | "left" | "right"
        self._evasion_action = None

    def get_actions(self, agent):
        sensor = agent.sensors[0]
        sensor.checkForLOSCollisions(agent.world)
        seen = sensor.agent_in_sight

        # if hunter still in view, either pick or continue evasion
        if seen is not None and getattr(seen, "team", None) != "runner":
            # first detection?
            if self._evasion_action is None:
                self._pick_new_evasion_action()
                self._sight_counter = 1
            else:
                # still evading—increment counter
                self._sight_counter += 1
                if self._sight_counter >= self.sight_threshold:
                    # re-pick a different maneuver
                    self._pick_new_evasion_action()
                    self._sight_counter = 1

            return self._do_evasion(self._evasion_action)

        # no hunter in sight → clear evasion and head to goal
        self._evasion_action = None
        self._sight_counter = 0
        return self._go_to_goal(agent)

    def _pick_new_evasion_action(self):
        r = random.random()
        if r < self.freeze_prob:
            self._evasion_action = "freeze"
        elif r < self.freeze_prob + self.dodge_left_prob:
            self._evasion_action = "left"
        else:
            self._evasion_action = "right"

    def _do_evasion(self, action):
        if action == "freeze":
            return 0.0, 0.0
        elif action == "left":
            return self.speed, +self.turn_rate
        else:  # "right"
            return self.speed, -self.turn_rate

    def _go_to_goal(self, agent):
        goal = np.asarray(self.goal_object.pos)
        diff = goal - agent.pos
        dist = np.linalg.norm(diff)
        if dist <= agent.radius:
            return 0.0, 0.0

        desired_angle = np.arctan2(diff[1], diff[0])
        v = self.speed
        omega = self.turn_rate if agent.angle < desired_angle else -self.turn_rate
        return v, omega

    @cached_property
    def goal_object(self):
        for obj in self.agent.world.objects:
            if isinstance(obj.name, str) and obj.name.lower() == self.target_name:
                return obj
        raise RuntimeError(f"Goal '{self.target_name}' not found")



    # @override
    # def __str__(self):
    #     return "SmartRunnerController"

    # def as_config_dict(self):
    #     return {'controller': self._config_controller}

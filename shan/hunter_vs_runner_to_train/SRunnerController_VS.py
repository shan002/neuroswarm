import numpy as np
from functools import cached_property
from swarmsim.agent.control.AbstractController import AbstractController

class SRunnerController_VS(AbstractController):
    """
    Runner that:
      • heads toward the goal when no hunter is seen
      • when a hunter is in view, splits its FOV in half, finds the closest hunter,
        and turns away: if hunter is on its left, turns right; if on its right, turns left.
    Also chooses a random speed within [low_threshold, speed] and holds it for change_interval steps.
    """

    def __init__(
        self,
        agent=None,
        parent=None,
        target_name="goal",
        speed=0.276,
        turn_rate=0.602,
        low_threshold=0.05,
        change_interval=50
    ):
        super().__init__(agent=agent, parent=parent)
        self.target_name = (
            target_name.lower() if isinstance(target_name, str) else target_name
        )
        self.speed = speed
        self.turn_rate = turn_rate
        # random-speed parameters
        self.low_threshold = low_threshold
        self.change_steps = change_interval
        # state for holding current speed
        self.steps_remaining = 0
        self.current_speed = self.speed

    def get_actions(self, agent):
        # 1) Update/hold current speed
        if self.steps_remaining <= 0:
            rng = getattr(agent, 'rng', np.random)
            self.current_speed = rng.uniform(self.low_threshold, self.speed)
            self.steps_remaining = self.change_steps
        self.steps_remaining -= 1

        # 2) Sense
        sensor = agent.sensors[0]
        sensor.checkForLOSCollisions(agent.world)
        seen = sensor.agent_in_sight

        # 3) Evade if hunter seen
        if seen is not None and getattr(seen, "team", None) != "runner":
            vec = np.asarray(seen.pos) - agent.pos
            angle_to_hunter = np.arctan2(vec[1], vec[0])
            rel = self._angle_diff(angle_to_hunter - agent.angle)
            omega = -self.turn_rate if rel > 0 else +self.turn_rate
            return self.current_speed, omega

        # 4) Otherwise, head toward goal
        return self._go_to_goal(agent)

    def _go_to_goal(self, agent):
        goal = np.asarray(self.goal_object.pos)
        diff = goal - agent.pos
        dist = np.linalg.norm(diff)
        if dist <= agent.radius:
            return 0.0, 0.0

        desired_angle = np.arctan2(diff[1], diff[0])
        rel = self._angle_diff(desired_angle - agent.angle)
        omega = self.turn_rate if rel > 0 else -self.turn_rate
        return self.current_speed, omega

    @cached_property
    def goal_object(self):
        for obj in self.agent.world.objects:
            if isinstance(obj.name, str) and obj.name.lower() == self.target_name:
                return obj
        raise RuntimeError(f"Goal '{self.target_name}' not found")

    @staticmethod
    def _angle_diff(a):
        """Normalize angle to [-π, +π]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

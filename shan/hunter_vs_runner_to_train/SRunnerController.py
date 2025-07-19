import numpy as np
from functools import cached_property
from swarmsim.agent.control.AbstractController import AbstractController

class SRunnerController(AbstractController):
    """
    Runner that:
      • heads toward the goal when no hunter is seen
      • when a hunter is in view, splits its FOV in half, finds the closest hunter,
        and turns away: if hunter is on its left, turns right; if on its right, turns left.
    """

    def __init__(
        self,
        agent=None,
        parent=None,
        target_name="goal",
        speed=0.276,
        turn_rate=0.602
    ):
        super().__init__(agent=agent, parent=parent)
        self.target_name = (
            target_name.lower() if isinstance(target_name, str) else target_name
        )
        self.speed = speed
        self.turn_rate = turn_rate

    def get_actions(self, agent):
        # 1) Sense
        sensor = agent.sensors[0]
        sensor.checkForLOSCollisions(agent.world)
        seen = sensor.agent_in_sight

        # 2) If hunter in view, compute evasion direction
        if seen is not None and getattr(seen, "team", None) != "runner":
            # vector from runner → hunter
            vec = np.asarray(seen.pos) - agent.pos
            angle_to_hunter = np.arctan2(vec[1], vec[0])
            # relative bearing in [-π, +π]
            rel = self._angle_diff(angle_to_hunter - agent.angle)

            # rel > 0 ⇒ hunter is on runner's left ⇒ turn RIGHT (negative ω)
            # rel < 0 ⇒ hunter on runner's right ⇒ turn LEFT (positive ω)
            omega = -self.turn_rate if rel > 0 else +self.turn_rate
            return self.speed, omega

        # 3) Otherwise, go to goal
        return self._go_to_goal(agent)

    def _go_to_goal(self, agent):
        goal = np.asarray(self.goal_object.pos)
        diff = goal - agent.pos
        dist = np.linalg.norm(diff)
        if dist <= agent.radius:
            return 0.0, 0.0

        desired_angle = np.arctan2(diff[1], diff[0])
        # turn toward desired_angle
        rel = self._angle_diff(desired_angle - agent.angle)
        omega = self.turn_rate if rel > 0 else -self.turn_rate
        return self.speed, omega

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

# TernaryFOVSensor.py

import pygame
import numpy as np
import math
from novel_swarms.sensors.AbstractSensor import AbstractSensor
from typing import List, TYPE_CHECKING, Any

from novel_swarms.world.goals.Goal import CylinderGoal

if TYPE_CHECKING:
    from novel_swarms.world.World import World
else:
    World = None

import warnings


class TernaryFOVSensor(AbstractSensor):
    """
    A Field‐of‐View sensor that distinguishes three cases:
      0 = no agent in FOV,
      1 = a fellow pursuer (hunter) in FOV,
      2 = the runner (evader) in FOV.
    """
    config_vars = AbstractSensor.config_vars + [
        'theta', 'distance', 'bias', 'false_positive', 'false_negative',
        'walls', 'wall_sensing_range', 'time_step_between_sensing',
        'invert', 'store_history', 'show'
    ]

    def __init__(
        self,
        agent=None,
        parent=None,
        theta: float = 10,
        distance: float = 100,
        bias: float = 0.0,
        false_positive: float = 0.0,
        false_negative: float = 0.0,
        walls=None,
        wall_sensing_range: float = 10,
        time_step_between_sensing: int = 1,
        invert: bool = False,
        store_history: bool = False,
        show: bool = True,
        seed: int | None = None,
        **kwargs: Any
    ):
        super().__init__(agent=agent, parent=parent)

        # Half‐angle of FOV cone (radians), sensing bias, FP/FN rates, etc.
        self.theta = theta
        self.bias = bias
        self.fp = false_positive
        self.fn = false_negative
        self.walls = walls
        self.wall_sensing_range = wall_sensing_range
        self.time_step_between_sensing = time_step_between_sensing
        self.time_since_last_sensing = 0
        self.store_history = store_history
        self.show = show
        self.invert = invert

        # Radius of agent sensing (for LOS checks)
        self.r = distance

        # History of detections (if store_history=True)
        self.history: list[int] = []

        # Random seed for FP/FN sampling
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # Detection bookkeeping
        self.agent_in_sight: Any = None
        self.current_state: int = 0   # values: 0 (none), 1 (hunter), 2 (runner)
        self.detection_id: int = 0

        # Handle deprecated "degrees" argument
        NOTFOUND = object()
        if (degrees := kwargs.pop('degrees', NOTFOUND)) is not NOTFOUND:
            warnings.warn("The 'degrees' kwarg is deprecated.", FutureWarning, stacklevel=1)
            if degrees:
                self.theta = np.radians(self.theta)

    def checkForLOSCollisions(self, world: World) -> None:
        """
        Perform a line‐of‐sight collision check against all agents and walls.
        If any agent is within the FOV cone (distance < r and within angle),
        select the closest intersecting agent. Otherwise, no detection.
        """
        self.time_since_last_sensing += 1
        if self.time_since_last_sensing % self.time_step_between_sensing != 0:
            return

        self.time_since_last_sensing = 0
        sensor_origin = self.agent.getPosition()

        # Collect all agents within radius r
        bag: list[Any] = []
        for other in world.population:
            if other is self.agent:
                continue
            if self.getDistance(sensor_origin, other.getPosition()) < self.r:
                bag.append(other)

        # Prepare FOV cone edges
        e_left, e_right = self.getSectorVectors()

        # Detect wall intersections (optional)
        consideration_set: list[tuple[float, Any]] = []
        if self.walls is not None:
            # Build line segments for FOV boundary rays
            l_seg = [sensor_origin, sensor_origin + (e_left[:2] * self.wall_sensing_range)]
            r_seg = [sensor_origin, sensor_origin + (e_right[:2] * self.wall_sensing_range)]
            wall_top    = [self.walls[0], [self.walls[1][0], self.walls[0][1]]]
            wall_right  = [[self.walls[1][0], self.walls[0][1]], self.walls[1]]
            wall_bottom = [self.walls[1], [self.walls[0][0], self.walls[1][1]]]
            wall_left   = [[self.walls[0][0], self.walls[1][1]], self.walls[0]]

            for wall in [wall_top, wall_right, wall_bottom, wall_left]:
                for line in [l_seg, r_seg]:
                    if self.lines_segments_intersect(line, wall):
                        d_to_inter = np.linalg.norm(
                            np.array(self.line_seg_int_point(line, wall)) - np.array(sensor_origin)
                        )
                        consideration_set.append((d_to_inter, None))

        # Check for agents in FOV cone
        for other in bag:
            u = other.getPosition() - sensor_origin
            d = self.circle_interesect_sensing_cone(u, self.agent.radius)
            if d is not None:
                consideration_set.append((d, other))

        if not consideration_set:
            # No candidate intersections ⇒ no detection
            self.determineState(False, None)
            return

        # Pick the closest intersection (lowest distance)
        consideration_set.sort(key=lambda x: x[0])
        _, detected_agent = consideration_set[0]
        self.determineState(True, detected_agent)

    def determineState(self, real_value: bool, agent: Any) -> None:
        """
        Given whether a detection occurred (real_value) and which agent (if any),
        set self.current_state ∈ {0,1,2} as follows:
          0 = no detection,
          1 = detected a fellow pursuer,
          2 = detected the runner.
        Applies false‐negative and false‐positive rates when appropriate.
        """
        if real_value:
            # Possibly flip to false negative
            if np.random.random_sample() < self.fn:
                # Record as "no detection"
                self.agent_in_sight = None
                self.current_state = 0 if not self.invert else 1
                self.detection_id = 0
            else:
                # A real intersection. Determine if it's the runner or a pursuer
                self.agent_in_sight = agent
                if agent is not None and getattr(agent, "team", None) == "runner":
                    # Runner‐in‐sight ⇒ state = 2
                    base_state = 2
                else:
                    # Another agent (pursuer) in sight ⇒ state = 1
                    base_state = 1
                self.current_state = base_state if not self.invert else (2 - base_state)
                self.detection_id = 0 if agent is None else agent.detection_id
        else:
            # No real detection. Possibly flip to false positive
            if np.random.random_sample() < self.fp:
                # False positive ⇒ claim "hunter in sight" (state = 1)
                self.agent_in_sight = None
                fake_state = 1
                self.current_state = fake_state if not self.invert else (2 - fake_state)
                self.detection_id = 0
            else:
                # Truly no detection
                self.agent_in_sight = None
                self.current_state = 0 if not self.invert else 2  # invert would treat "no detection" as 2
                self.detection_id = 0

        # Optionally record history
        if self.store_history:
            self.history.append(self.current_state)

    def step(self, world: World, only_check_goals: bool = False) -> None:
        super(TernaryFOVSensor, self).step(world=world)
        # We do not separately check goals here, since TernaryFOVSensor focuses on agents only
        self.checkForLOSCollisions(world=world)

    def draw(self, screen, offset=((0, 0), 1.0)):
        super(TernaryFOVSensor, self).draw(screen, offset)
        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
        if self.show:
            # Choose color based on current_state: 0=red (no sight), 1=green (hunter), 2=yellow (runner)
            if self.current_state == 0:
                sight_color = (255, 0, 0)
            elif self.current_state == 1:
                sight_color = (0, 255, 0)
            else:  # self.current_state == 2
                sight_color = (255, 255, 0)

            magnitude = self.r if self.agent.is_highlighted else self.agent.radius * 5
            head = np.asarray(self.agent.getPosition()) * zoom + pan
            e_left, e_right = self.getSectorVectors()
            e_left, e_right = np.asarray(e_left[:2]), np.asarray(e_right[:2])

            tail_l = head + magnitude * e_left * zoom
            tail_r = head + magnitude * e_right * zoom

            pygame.draw.line(screen, sight_color, head, tail_l)
            pygame.draw.line(screen, sight_color, head, tail_r)
            if self.agent.is_highlighted:
                width = max(1, round(0.01 * zoom))
                pygame.draw.circle(screen, sight_color + (50,), head, self.r * zoom, width)
                if self.wall_sensing_range:
                    pygame.draw.circle(
                        screen,
                        (150, 150, 150, 50),
                        head,
                        self.wall_sensing_range * zoom,
                        width
                    )

    def circle_interesect_sensing_cone(self, u: np.ndarray, r: float) -> float | None:
        """
        Check whether a circle of radius r (centered at vector u from sensor origin)
        intersects the FOV cone defined by edges e_left and e_right.
        Returns the distance to intersection if it exists, else None.
        """
        e_left, e_right = self.getSectorVectors()
        directional = np.dot(u, self.getBiasedSightAngle())
        if directional > 0:
            u = np.append(u, [0])
            cross_l = np.cross(e_left, u)
            cross_r = np.cross(u, e_right)
            sign_l = np.sign(cross_l)
            sign_r = np.sign(cross_r)
            added_signs = sign_l - sign_r
            if np.all(added_signs == 0):
                return np.linalg.norm(u[:2])

            # Check circle vs. cone edge intersections
            u_l = np.dot(u, e_left) * e_left
            u_r = np.dot(u, e_right) * e_right
            dist_l = np.linalg.norm(u - u_l)
            dist_r = np.linalg.norm(u - u_r)
            if dist_l < r or dist_r < r:
                return np.linalg.norm(u[:2])
        return None

    def getDistance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(b - a)

    def getLOSVector(self) -> List[float]:
        head = self.agent.getPosition()
        tail = self.getFrontalPoint()
        return [tail[0] - head[0], tail[1] - head[1]]

    def getFrontalPoint(self) -> np.ndarray:
        if self.angle is None:
            return self.agent.getFrontalPoint()
        return self.agent.pos + [
            math.cos(self.angle + self.agent.angle),
            math.sin(self.angle + self.agent.angle)
        ]

    def getBiasedSightAngle(self) -> np.ndarray:
        bias_transform = np.array([
            [np.cos(self.bias), -np.sin(self.bias), 0],
            [np.sin(self.bias),  np.cos(self.bias), 0],
            [0,                  0,                 1]
        ])
        v = np.append(self.getLOSVector(), [0])
        return np.matmul(bias_transform, v)[:2]

    def getSectorVectors(self) -> tuple[np.ndarray, np.ndarray]:
        theta_l = self.theta + self.bias
        theta_r = -self.theta + self.bias
        rot_z_left = np.array([
            [np.cos(theta_l), -np.sin(theta_l), 0],
            [np.sin(theta_l),  np.cos(theta_l), 0],
            [0,                0,               1]
        ])
        rot_z_right = np.array([
            [np.cos(theta_r), -np.sin(theta_r), 0],
            [np.sin(theta_r),  np.cos(theta_r), 0],
            [0,                0,               1]
        ])
        v = np.append(self.getLOSVector(), [0])
        e_left  = np.matmul(rot_z_left,  v)
        e_right = np.matmul(rot_z_right, v)
        return e_left, e_right

    def lines_segments_intersect(self, l1: list[list[float]], l2: list[list[float]]) -> bool:
        p1, q1 = l1
        p2, q2 = l2
        o1 = self.point_orientation(p1, q1, p2)
        o2 = self.point_orientation(p1, q1, q2)
        o3 = self.point_orientation(p2, q2, p1)
        o4 = self.point_orientation(p2, q2, q1)
        return (o1 != o2) and (o3 != o4)

    def line_seg_int_point(self, line1: list[list[float]], line2: list[list[float]]) -> tuple[float, float]:
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def point_orientation(self, p1: list[float], p2: list[float], p3: list[float]) -> int:
        val = ((p2[1] - p1[1]) * (p3[0] - p2[0])) - ((p2[0] - p1[0]) * (p3[1] - p2[1]))
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0

    def as_config_dict(self) -> dict[str, Any]:
        return {
            "type": "TernaryFOVSensor",
            "theta": self.theta,
            "bias": self.bias,
            "fp": self.fp,
            "fn": self.fn,
            "time_step_between_sensing": self.time_step_between_sensing,
            "store_history": self.store_history,
            "wall_sensing_range": self.wall_sensing_range,
            "agent_sensing_range": self.r,
            "seed": self.seed if self.seed is not None else None,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TernaryFOVSensor":
        return TernaryFOVSensor(
            agent=None,
            theta=d["theta"],
            distance=d["agent_sensing_range"],
            bias=d["bias"],
            false_positive=d.get("fp", 0.0),
            false_negative=d.get("fn", 0.0),
            store_history=d["store_history"],
            wall_sensing_range=d["wall_sensing_range"],
            time_step_between_sensing=d["time_step_between_sensing"],
            seed=d.get("seed", None),
        )

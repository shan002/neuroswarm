import math
import random
import argparse
import gym
import turtle
import time
from turtle import Vec2D
from functools import cached_property
from dataclasses import dataclass
# turtle.ht()

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=10)
parser.add_argument('--net')
parser.add_argument('--seed', default=None)
parser.add_argument('--fps', default=None)
parser.add_argument('--render_mode',
                    choices=['None', 'human', 'rgb_array'], default='human',
                    help="use `human` to show or `None` to hide. Default `human`.")

DEFAULT_N = 10


def rf(factor=1, randomizer=random):
    return randomizer.random() * factor


def rif(factor=1, randomizer=random):
    return (randomizer.random() * 2 - 1) * factor


def is_circle_outside_bounds(x1, y1, x2, y2, x, y, r) -> bool:
    return x - r < x1 or x + r > x2 or y - r < y1 or y + r > y2


def is_point_outside_bounds(x1, y1, x2, y2, x, y) -> bool:
    return x < x1 or x > x2 or y < y1 or y > y2


class RateLimiter:
    def __init__(self, hz=None, period=None):
        if hz is not None:
            self.hz = hz
        elif period is not None:
            self.period = period
        self.last = None

    @property
    def hz(self):
        return 1 / self.period

    @hz.setter
    def hz(self, x):
        self.period = 1 / x

    def wait(self):
        now = time.time()
        try:
            diff = now - self.last
        except TypeError:
            pass  # self.last was not set. Skip check this time, goto finally
        else:
            if diff < self.period:
                time.sleep(self.period - diff)
        finally:
            self.last = now


class FOV:
    def __init__(self, agent, d, fov=None, l=None, r=None):  # noqa
        self.a = agent
        self.d = d
        self.set_fov(fov, l, r)
        self.activated = None

    def set_fov(self, fov=None, l=None, r=None):  # noqa
        if fov is not None:
            self.fov = fov
            self.left = 0 if fov == 0 else self.fov / 2
            self.right = 0 if fov == 0 else -self.fov / 2
            if l is not None or r is not None:
                raise ValueError("Only fov OR l/r stops can be specified.")
        else:
            self.left = l
            self.right = r
            self.fov = abs(l - r)

        if self.fov < 0:
            raise ValueError("fov must be nonnegative")

    def __contains__(self, b):
        if self.a == b:
            self.activated = False
            return self.activated
        within_range = self.a.distance(b) < self.d
        theta = self.a.towards(b) - self.a.heading()
        theta = (theta + 180) % 360 - 180
        within_fov = self.left > theta > self.right
        self.activated = within_range and within_fov
        return self.activated

    def whisker(self, angle):
        x, y = self.a.position()
        theta = self.a.heading() + angle
        dx = self.d * math.cos(math.radians(theta))
        dy = self.d * math.sin(math.radians(theta))
        return (x, y), (x + dx, y + dy)

    @property
    def lwhisker(self):
        return self.whisker(self.left)

    @property
    def rwhisker(self):
        return self.whisker(self.right)


class TurtleShell(turtle.TNavigator):
    """
    Turtle class providing compatibility between RawTurtle and TNavigator.

    This class allows me to use TNavigator instead of RawTurtle
    It patches over some methods in RawTurtle
    So I can prevent any graphical/ui/TKinter code from running when unneeded

    # NB: Degrees are the canonical unit used in turtle.py
    #     but turtle.py also has a radians mode. Don't turn that on.
    """

    def __init__(self, *args, **kwargs):
        if isinstance(self, turtle.RawTurtle):
            super().__init__(*args, **kwargs)
        else:
            super().__init__()
        self.speed(0)

        self.ir_region = FOV(self, d=3.2, l=12, r=4)
        self.camera_region = FOV(self, d=5, fov=54)

        self.r = 0.16

    def penup(self):
        # RawTurtle has penup() but TNavigator does not. Need to prevent error on call to penup()
        if isinstance(self, turtle.RawTurtle):
            super().penup()

    def clear(self):
        if isinstance(self, turtle.RawTurtle):
            super().clear()


class Turtle(TurtleShell, turtle.RawTurtle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penup()
        self.clear()


class Sim(gym.Env):
    metadata = {"render_modes": ["None", "human", "rgb_array"], }

    def __init__(
        self,
        n=10,
        randomizer=random,
        render_mode=None,
        world_size=(15, 15),
        tps=7.5,
        fps_limit=None,
    ):
        super().__init__()
        self.n = n
        self.render_mode = render_mode
        self._renderer = None
        self.screen = None
        self.world_size = world_size
        self.tps = tps
        self.tsync = RateLimiter(hz=fps_limit if fps_limit is not None else tps)

        self.pre_callback = None
        self.post_callback = None

        self.randomizer = randomizer

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode in ("human", "rgb_array"):
            self.RobotClass = Turtle
            self.screen = turtle._Screen()
            self._renderer = "turtle"
        else:
            self.RobotClass = TurtleShell
            self._renderer = None

        self.reset()

    def reset(self, seed=None):
        if self.seed is not None:
            self.randomizer.seed(seed)

        w, h = self.world_size

        # clear turtle.py default environment and stop autostepping
        if self._renderer == "turtle":
            self.screen.clear()
            self.screen.clear()  # grr. buggy?
            self.screen.tracer(0, 0)
            self.set_window_shape(600, 600)
            self.screen.setworldcoordinates(*self.world_bounds)
            self.pen = self.RobotClass(self.screen, undobuffersize=0)

        # goal setup
        self.goal = self.RobotClass(self.screen, undobuffersize=0)
        self.goal.r = 3
        self.goal.setpos(0, 0)
        self.goal.setpos(2.5, 2.5)

        # per-robot/turtle setup
        self.robots = [self.RobotClass(self.screen, undobuffersize=0) for i in range(self.n)]
        for robot in self.robots:  # WARNING: monkey patching! consider moving these props to TurtleShell
            robot.visited_goal = 0
            robot.was_on_goal = False
        if self._renderer == "turtle":
            self.screen.tracer(0, 0)
            # self.draw_circle(*self.goal.pos, self.goal.r, fill='green', outline="")
            # self.goal.turtlesize(self.goal.r * 6, self.goal.r * 6)
            self.draw_world_bounds()
            self.draw_goal()
        for robot in self.robots:
            robot.setposition(rif(1, self.randomizer) - 2, rif(1, self.randomizer) - 2)
            robot.setheading(rf(360, self.randomizer))
        self.accumulator = 0
        self.tick = 0

    @cached_property
    def world_bounds(self):
        w, h = self.world_size
        return (-w / 2, -h / 2, w / 2, h / 2)

    def draw_world_bounds(self):
        x1, y1, x2, y2 = self.world_bounds
        self.pen.setposition(x2, y2)
        self.pen.pendown()
        for x, y in ((x2, y1), (x1, y1), (x1, y2), (x2, y2)):
            self.pen.setposition(x, y)
        self.pen.hideturtle()

    def draw_goal(self):
        x, y = self.goal.position()
        r = self.goal.r
        self.goal.setheading(0)
        self.goal.setposition(x, y - r)
        self.goal.shape("circle")
        self.goal.color("green", "#99FF99")
        self.goal.begin_fill()
        self.goal.pendown()
        self.goal.circle(r)
        self.goal.end_fill()
        self.goal.penup()
        self.goal.setposition(x, y)

    def set_window_shape(self, w, h):
        self.screen.getcanvas().master.geometry(f"{w}x{h}")

    def seed(self, new_seed):
        self.seed = new_seed

    # def draw_circle(self, x, y, r, **kwargs):
    #     # https://stackoverflow.com/a/17985217/2712730
    #     return self.screen.cv.create_oval(x - r, y - r, x + r, y + r, **kwargs)

    def global_centroid(self):
        xsum = ysum = 0
        for robot in self.robots:
            xsum += robot.xcor()
            ysum += robot.ycor()
        xsum /= self.n
        ysum /= self.n
        return xsum, ysum

    def circleness(self):
        center = self.global_centroid()
        rin, rout = float('inf'), 0
        for robot in self.robots:
            r = robot.distance(center)
            rin = r if r < rin else rin
            rout = r if r > rout else rout
        return rin - rout

    @classmethod
    def dynamics(cls, x):
        # wl, wr, theta = x
        # r = 0.020  # meters
        # wb = 0.150  # meters
        # v = r * (wl + wr) / 2

        # dx = v * cos(math.radians(theta))
        # dy = v * sin(math.radians(theta))
        # dtheta = r / 2 / wb * (wl - wr)
        v, w, theta = x
        dx = v * math.cos(math.radians(theta))
        dy = v * math.sin(math.radians(theta))
        dtheta = math.degrees(w)

        return dx, dy, dtheta

    def step(self, action):
        if self.pre_callback is not None:
            self.pre_callback(self, action)

        div = self.tps

        # Move robots according to actions
        for robot, act in zip(self.robots, action):
            v, w = act
            x, y = robot.position()
            theta = robot.heading()
            r = robot.r
            dx, dy, dtheta = Sim.dynamics((v, w, theta))
            dx, dy, dtheta = dx / div, dy / div, dtheta / div

            # temporarily move robot to check for collisions
            robot.setposition(x + dx, y + dy)
            robot.setheading(theta + dtheta)

            # DO COLLISION DETECTION HERE PLS
            nearest_neighbor = min([i for i in self.robots if i is not robot], key=robot.distance)
            collided_with_agent = robot.distance(nearest_neighbor) < r + nearest_neighbor.r
            facing_eachother = abs(nearest_neighbor.heading() - robot.heading() - 180) % 360 < 60

            collided_with_wall = is_circle_outside_bounds(*self.world_bounds, *robot.position(), robot.r)

            if (
                (collided_with_agent and facing_eachother)
                or collided_with_wall
            ):
                # undo move
                robot.setposition(x, y)
                robot.setheading(theta)

        # if action:
        #     for robot, act in zip(self.robots, action):
        #         if act:
        #             robot.left(self.turning_rate)
        #         else:
        #             robot.right(self.turning_rate)
        #         robot.forward(self.speed)  # always move forwards

        qty_on_goal = 0

        # n-length list[tuple] of sensor output of each robot
        def query_sensor(robot):
            nonlocal qty_on_goal
            # WARNING: This is not a pure function!

            # infrared/proximity sensor detections
            can_see_other_robot = any(other in robot.ir_region for other in self.robots if other is not robot)
            can_see_goal = self.goal in robot.camera_region
            on_goal = robot.distance(self.goal) < self.goal.r
            can_see_wall = (is_point_outside_bounds(*self.world_bounds, *robot.ir_region.lwhisker[1])
                            or is_point_outside_bounds(*self.world_bounds, *robot.ir_region.rwhisker[1]))

            sensor_triggered = can_see_other_robot or can_see_wall

            if on_goal:
                # if the robot has just entered the goal on this tick
                qty_on_goal += 1
                if not robot.was_on_goal:
                    robot.visited_goal += 1
            #         self.accumulator -= 0.1
            #         if robot.visited_goal == 1:
            #             # reward for visiting goal for the first time
            #             self.accumulator += 1000

            if self._renderer == "turtle":
                if sensor_triggered:
                    robot.fillcolor("red")
                elif can_see_goal:
                    robot.fillcolor("#33dd33")
                else:
                    robot.fillcolor("black")
                if on_goal:
                    robot.pencolor("#ff2211")
                elif robot.visited_goal:
                    robot.pencolor("green")

            # save state to prior
            # robot.was_on_goal = on_goal
            return (sensor_triggered, can_see_goal)

        observations = [query_sensor(robot) for robot in self.robots]

        # how_many_visited_goal = sum([bool(robot.visited_goal) for robot in self.robots])
        # punish the longer we go without more robots visiting goal for first time
        # self.accumulator -= self.n - how_many_visited_goal

        self.tick += 1
        return observations, qty_on_goal, False, False, self.tick - 1, None, False
        # observation:object, reward:float, terminated:bool, truncated:bool, info:dict, deprecated, done:bool

    def render(self):
        if self._renderer == "turtle":
            self.screen.update()
            self.tsync.wait()


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed()

    try:
        from neuro import MOAna
        rnd = MOA()
        rnd.seed(args.seed if args.seed else random.randrange(32767))
    except (ModuleNotFoundError, ImportError):
        print("neuro.MOA not found. Using python random")
        rnd = random.Random()
        rnd.seed(args.seed)

    sim = Sim(
        args.n,
        randomizer=rnd,
        render_mode=args.render_mode
    )
    # actions = [None] * args.n
    actions = [(0, 0)] * args.n
    for _ in range(3000):
        observations, reward, *_ = sim.step(actions)
        actions = [(0.150, 0.5 if ir_sensed else -0.5) for ir_sensed, cam_sensed in observations]
        sim.render()
    print(reward)

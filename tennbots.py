import math
import turtle
from turtle import Vec2D
# import tkinter as TK
import random
import argparse
import gym
# turtle.ht()

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=10)
parser.add_argument('--net')
parser.add_argument('--seed', default=None)
parser.add_argument('--render_mode',
                    choices=['None', 'human', 'rgb_array'], default='human',
                    help="use `human` to show or `None` to hide. Default `human`.")

DEFAULT_N = 10
DEFAULT_FOV = 12
DEFAULT_RANGE = 4
DEFAULT_TURNINGRATE = 7.8
DEFAULT_SPEED = 0.5


def rf(factor=1, randomizer=random):
    return randomizer.random() * factor


def rif(factor=1, randomizer=random):
    return (randomizer.random() * 2 - 1) * factor


class TurtleShell(turtle.TNavigator):
    """
    Turtle class providing compatibility between RawTurtle and TNavigator.

    This class allows me to use TNavigator instead of RawTurtle
    It patches over some methods in RawTurtle
    So I can prevent any graphical/ui/TKinter code from running when unneeded
    """

    def __init__(self, *args, **kwargs):
        if isinstance(self, turtle.RawTurtle):
            super().__init__(*args, **kwargs)
        else:
            super().__init__()
        self.sensor_range = DEFAULT_RANGE
        self.sensor_fov = DEFAULT_FOV

    def penup(self):
        # RawTurtle has penup() but TNavigator does not. Need to prevent error on call to penup()
        if isinstance(self, turtle.RawTurtle):
            super().penup()

    def clear(self):
        if isinstance(self, turtle.RawTurtle):
            super().clear()

    def can_see(self, other):
        if self == other:
            return False
        within_range = self.distance(other) < self.sensor_range
        h1 = self.heading()
        h2 = self.towards(other)
        theta = h2 - h1
        within_fov = abs(theta) < self.sensor_fov / 2
        return within_range and within_fov


class Turtle(TurtleShell, turtle.RawTurtle):
    pass


class Sim(gym.Env):
    metadata = {"render_modes": ["None", "human", "rgb_array"], }

    def __init__(
        self,
        n=DEFAULT_N,
        fov=DEFAULT_FOV,
        vsn_range=DEFAULT_RANGE,
        render_mode=None,
        randomizer=random,
    ):
        super().__init__()
        self.n = n
        self.fov = fov
        self.range = vsn_range
        self.speed = DEFAULT_SPEED
        self.turning_rate = DEFAULT_TURNINGRATE
        self.render_mode = render_mode
        self._renderer = None
        self.screen = None

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

        self.reset(self.seed)

    class Goal():
        def __init__(self, pos, r):
            self.pos = Vec2D(*pos)
            self.r = r

    def reset(self, seed=None):
        if self.seed is not None:
            self.randomizer.seed(seed)

        # clear turtle.py default environment and stop autostepping
        if self._renderer == "turtle":
            self.screen.clear()
            self.screen.clear()  # grr. buggy?
            self.screen.tracer(0, 0)

        self.goal = self.Goal((10, 10), 1)
        # per-turtle setup
        self.robots = [self.RobotClass(self.screen, undobuffersize=0) for i in range(self.n)]
        for robot in self.robots:
            robot.visited_goal = 0
            robot.was_on_goal = False
        if self._renderer == "turtle":
            self.screen.tracer(0, 0)
            self.draw_circle(*self.goal.pos, self.goal.r, fill='green', outline="")
        for robot in self.robots:
            robot.speed(0)
            robot.penup()
            robot.clear()
            robot.setposition(rif(2, self.randomizer), rif(2, self.randomizer))
            robot.setheading(rf(360, self.randomizer))
        self.accumulator = 0
        self.tick = 0

    def seed(self, new_seed):
        self.seed = new_seed

    def draw_circle(self, x, y, r, **kwargs):
        # https://stackoverflow.com/a/17985217/2712730
        return self.screen.cv.create_oval(x - r, y - r, x + r, y + r, **kwargs)

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
        wl, wr, theta = x
        cos, sin = math.cos, math.sin
        r = 0.020  # meters
        wb = 0.150  # meters
        v = r * (wl + wr) / 2

        dx = v * cos(math.radians(theta))
        dy = v * sin(math.radians(theta))
        dtheta = r / 2 / wb * (wl - wr)

        return dx, dy, dtheta

    def step(self, action):
        # Move robots according to actions
        for robot, act in zip(self.robots, action):
            wl, wr = act
            x, y = robot.position()
            theta = robot.heading()
            dx, dy, dtheta = Sim.dynamics((wl, wr, theta))

            robot.setposition(x + dx, y + dy)
            robot.setheading(theta + dtheta)

        # if action:
        #     for robot, act in zip(self.robots, action):
        #         if act:
        #             robot.left(self.turning_rate)
        #         else:
        #             robot.right(self.turning_rate)
        #         robot.forward(self.speed)  # always move forwards

        # n-length list[tuple] of sensor output of each robot
        def query_sensor(robot):
            # WARNING: This is not a pure function!

            # infrared/proximity sensor detections
            can_see_other_robot = any(robot.can_see(other) for other in self.robots if other is not robot)
            can_see_goal = robot.can_see(self.goal.pos)  # assumes infrared detection range is same for goal
            sensor_triggered = can_see_other_robot or can_see_goal

            on_goal = robot.distance(self.goal.pos) < self.goal.r

            if on_goal:
                # if the robot has just entered the goal on this tick
                if not robot.was_on_goal:
                    robot.visited_goal += 1
                    self.accumulator -= 0.1
                    if robot.visited_goal == 1:
                        # reward for visiting goal for the first time
                        self.accumulator += 1000

            if self._renderer == "turtle":
                if sensor_triggered:
                    robot.fillcolor("red")
                else:
                    robot.fillcolor("black")
                if robot.visited_goal:
                    robot.fillcolor("green")

            # save state to prior
            robot.was_on_goal = on_goal
            return (sensor_triggered, on_goal)

        observations = [query_sensor(robot) for robot in self.robots]

        how_many_visited_goal = sum([bool(robot.visited_goal) for robot in self.robots])
        # punish the longer we go without more robots visiting goal for first time
        self.accumulator -= self.n - how_many_visited_goal

        self.tick += 1
        return observations, self.accumulator, False, False, False, None, False
        # observation:object, reward:float, terminated:bool, truncated:bool, info:dict, deprecated, done:bool

    def render(self):
        if self._renderer == "turtle":
            self.screen.update()


if __name__ == "__main__":
    args = parser.parse_args()

    # with open(args.network_file, 'r') as f:
    #     json_str = f.read()

    # net = neuro.Network()
    # net.from_str(json_str)

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
    while(True):
        observations, reward, *_ = sim.step(actions)
        actions = observations
        sim.render()
        print(reward)

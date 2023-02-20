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
DEFAULT_FOV = 40
DEFAULT_RANGE = 90
DEFAULT_TURNINGRATE = 7.8
DEFAULT_SPEED = 0.5


def rf(factor=1, randomizer=random):
    return randomizer.random() * factor


def rif(factor=1, randomizer=random):
    return (randomizer.random() * 2 - 1) * factor


class TurtleShell(turtle.TNavigator):
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

    def can_see(self, other):
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

    def reset(self, seed=None):
        if self.seed is not None:
            self.randomizer.seed(seed)
        if self._renderer == "turtle":
            self.screen.clear()
            self.screen.tracer(0, 0)
        self.robots = [self.RobotClass(self.screen, undobuffersize=0) for i in range(self.n)]
        # per-turtle setup
        for robot in self.robots:
            robot.speed(0)
            robot.setposition(rif(10, self.randomizer), rif(10, self.randomizer))
            robot.setheading(rf(360, self.randomizer))
            robot.penup()

    def seed(self, new_seed):
        self.seed = new_seed

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

    def step(self, action):
        if action:
            for robot, act in zip(self.robots, action):
                if act:
                    robot.left(self.turning_rate)
                else:
                    robot.right(self.turning_rate)
                robot.forward(self.speed)  # always move forwards

        # n-length list[bool] of whether each robot sees any other robots
        observations = [any([robot.can_see(other) for other in self.robots if other is not robot]) for robot in self.robots]

        return observations, self.circleness(), False, False, None, None, False
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
    actions = [None] * args.n
    while(True):
        observations, reward, *_ = sim.step(actions)
        actions = observations
        sim.render()
        print(reward)

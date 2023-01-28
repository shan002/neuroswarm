import turtle
# import tkinter as TK
import random
import argparse
# turtle.ht()

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=10)
parser.add_argument('--fov', default=40)
parser.add_argument('--range', default=200)
parser.add_argument('--turning_rate', default=7.8)
parser.add_argument('--speed', default=0.5)


DEFAULT_N = 10
DEFAULT_FOV = 40
DEFAULT_RANGE = 200
DEFAULT_TURNINGRATE = 7.8
DEFAULT_SPEED = 0.5

# uses system time if empty
random.seed()


def rf(factor=1):
    return random.random() * factor


def rif(factor=1):
    return (random.random() * 2 - 1) * factor


class Robot(turtle.RawTurtle):
    def can_see(self, other, fov):
        within_range = self.distance(other) < 90
        h1 = self.heading()
        h2 = self.towards(other)
        theta = h2 - h1
        within_fov = abs(theta) < fov / 2
        return within_range and within_fov


class Run():
    def __init__(
        self,
        n=DEFAULT_N,
        fov=DEFAULT_FOV,
        vsn_range=DEFAULT_RANGE,
        turning_rate=DEFAULT_TURNINGRATE,
        speed=DEFAULT_SPEED,
        visible=True,
        screen=None,
    ):
        self.n = n
        self.fov = fov
        self.range = vsn_range
        self.turning_rate = turning_rate
        self.speed = speed
        self.visible = visible

        if screen is None:
            self.screen = screen = turtle.Screen()
            turtle.TurtleScreen._RUNNING = True
        self.robots = [Robot(screen, undobuffersize=0) for i in range(n)]
        screen.tracer(0, 0)
        # per-turtle setup
        for robot in self.robots:
            robot.penup()
            robot.speed(0)
            robot.setposition(rif(10), rif(10))
            robot.setheading(rf(360))

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

    def step(self):
        for robot in self.robots:
            see = False
            # check if this robot can see any other robots
            for other in self.robots:
                if robot == other:
                    continue  # skip check on self
                if robot.can_see(other, self.fov):
                    see = True

            robot.left(0 if see else self.turning_rate)
            robot.forward(self.speed)
        if self.visible:
            self.screen.update()


if __name__ == "__main__":
    args = parser.parse_args()

    batch = Run(
        args.n,
        args.fov,
        args.range,
        args.turning_rate,
        args.speed,
    )

    while(True):
        batch.step()
        print(batch.circleness())

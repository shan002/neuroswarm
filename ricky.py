import turtle
# import tkinter as TK
import random
import argparse
# turtle.ht()

DEFAULT_N = 10
DEFAULT_FOV = "40"
DEFAULT_RANGE = "90"
DEFAULT_TURNINGRATE = "7.8"
DEFAULT_SPEED = "0.5"

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=DEFAULT_N)
parser.add_argument('--fov', default=DEFAULT_FOV)
parser.add_argument('--range', default=DEFAULT_RANGE)
parser.add_argument('--turning_rate', default=DEFAULT_TURNINGRATE)
parser.add_argument('--speed', default=DEFAULT_SPEED)


# uses system time if empty
random.seed()


def rf(factor=1):
    return random.random() * factor


def rif(factor=1):
    return (random.random() * 2 - 1) * factor


class Robot(turtle.RawTurtle):
    def can_see(self, other, fov):
        if self == other:
            return False
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
        self.n = int(n)
        self.fov = eval(fov)
        self.range = eval(vsn_range)
        self.turning_rate = eval(turning_rate)
        self.speed = eval(speed)
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

    def circle_size(self):
        center = self.global_centroid()
        rin, rout = float('inf'), 0
        for robot in self.robots:
            r = robot.distance(center)
            rin = r if r < rin else rin
            rout = r if r > rout else rout
        return rout

    def step(self):
        for robot in self.robots:
            see = False
            # check if this robot can see any other robots
            for other in self.robots:
                if robot == other:
                    continue  # skip check on self
                if robot.can_see(other, self.fov):
                    see = True

        observations = [any(robot.can_see(other, self.fov) for other in self.robots) for robot in self.robots]
        for see, robot in zip(observations, self.robots):
            robot.left(-self.turning_rate if see else self.turning_rate)
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
        print(f"circlyness: {batch.circleness(): 9.3f}\t r: {batch.circle_size(): 9.3f}")
        # print(batch.circleness())

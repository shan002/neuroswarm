from swarmsim.metrics.AbstractMetric import AbstractMetric

class CatchRunnerMetricWTP_DPC(AbstractMetric):
    def __init__(self, name="CatchRunnerMetric", history_size=1):
        super().__init__(name=name, history_size=history_size)
        self.detect_runner = False

    def attach_world(self, world):
        self.world = world

    def check_if_runner_in_sight(self, agent):
        agent.sensors[0].checkForLOSCollisions(self.world) # this line is needed to create agent_in_sight attribute for BinaryFOVSensor
        a_s = agent.sensors[0].agent_in_sight
        return a_s != None and a_s.team == "runner"
    
    def check_sim_result(self) -> str | None:
        """
        Returns:
        - "Success" if runner is caught by any hunter
        - "Failure" if runner reaches goal
        - None if neither has occurred yet
        """
        import numpy as np
        COLLISION_CLEARANCE = 0.005

        runner = next((agent for agent in self.world.population if getattr(agent, "team", None) == "runner"), None)
        if runner is None:
            return None

        goal = next((obj for obj in self.world.objects if isinstance(obj.name, str) and obj.name.lower() == "goal"), None)
        
        if goal is not None:
            goal_radius = getattr(goal, "agent_radius", 0.2)
            if np.linalg.norm(np.array(runner.pos) - np.array(goal.pos)) <= (runner.radius + goal_radius): # distance 
                return "Failure"

        for agent in self.world.population:
            if getattr(agent, "team", None) != "runner":
                if self.check_if_runner_in_sight(agent):
                    self.detect_runner = True

                if np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + COLLISION_CLEARANCE):
                    return "Success"

        return None

    def calculate(self):
        result = self.check_sim_result()
        result_step = self.world.total_steps
        total_steps_to_finish = self.world.config.stop_at
        if result == "Success":
            reward = 1.0 - (result_step / total_steps_to_finish)
            self.set_value(reward)
        elif result == "Failure":
            self.set_value(0.0)
        elif result_step == total_steps_to_finish:
            if self.detect_runner:
                self.set_value(0.2) # Giving partial credit if the hunters can detect the runner
                self.detect_runner = False
            else:
                self.set_value(0.1) # the runner couldn't reach the goal within the given time so it's kind of a win in some way








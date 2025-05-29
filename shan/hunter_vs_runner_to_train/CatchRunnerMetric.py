from novel_swarms.metrics.AbstractMetric import AbstractMetric

class CatchRunnerMetric(AbstractMetric):
    def __init__(self, name="CatchRunnerMetric", history_size=1):
        super().__init__(name=name, history_size=history_size)

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
                if self.check_if_runner_in_sight(agent) or np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + COLLISION_CLEARANCE):
                # if np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + COLLISION_CLEARANCE):
                    return "Success"

        return None

    def calculate(self):
        result = self.check_sim_result()
        if result == "Success":
            self.set_value(1.0)
        elif result == "Failure":
            self.set_value(0.0)
        else:
            self.set_value(None)




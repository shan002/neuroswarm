from novel_swarms.metrics.AbstractMetric import AbstractMetric
from catch_runner_check_result import check_sim_result

class CatchRunnerMetric(AbstractMetric):
    def __init__(self, name="CatchRunnerMetric", history_size=1):
        super().__init__(name=name, history_size=history_size)

    def attach_world(self, world):
        self.world = world

    def calculate(self):
        result = self.world.meta.get("result", check_sim_result(self.world))  # fallback if not stored
        result_step = self.world.total_steps
        # print(result_step)
        if result == "Success":
            # print(self.world.config.stop_at)
            reward = 1.0 #- (result_step / self.world.config.stop_at)
            self.set_value(reward)
        elif result == "Failure":
            self.set_value(0.0)
        else:
            self.set_value(None)




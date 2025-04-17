import numpy as np

def check_sim_result(world):
    # Find the runner agent tagged with team "runner"
    runner = next((agent for agent in world.population if getattr(agent, "team", None) == "runner"), None)
    if runner is None:
        return None

    # Find the goal object with the name "goal"
    goal = next((obj for obj in world.objects if isinstance(obj.name, str) and obj.name.lower() == "goal"), None)
    
    # Check if the runner reached the goal first; if so, it's a failure for the hunters.
    if goal is not None:
        goal_size = getattr(goal, "size", 0.2)
        if np.linalg.norm(np.array(runner.pos) - np.array(goal.pos)) <= (runner.radius + goal_size):
            return "Failure"
    
    # Only if the runner hasn't reached the goal, check if any hunter touches the runner.
    for agent in world.population:
        if getattr(agent, "team", None) != "runner":
            if np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + 0.005):
                return "Success"
    
    # No event detected yet
    return None
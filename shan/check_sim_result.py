import numpy as np

def check_sim_result(world):
    # Find the runner agent tagged with team "runner"
    runner = next((agent for agent in world.population if getattr(agent, "team", None) == "runner"), None)
    if runner is None:
        return None

    # Find the goal object with the name "goal")
    goal = next((obj for obj in world.objects if isinstance(obj.name, str) and obj.name.lower() == "goal"), None)

    result = None

    # Check collision with any hunter
    for agent in world.population:
        if getattr(agent, "team", None) != "runner":
            if np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + 0.005): # Just giving a clearance of 0.005
                result = "Success"

    goal_size = getattr(goal, "size", 0.2) if goal else 0.2 ## Need to update this. this doesn't reflect the size of the actual goal.

    # Check if runner reached the goal using the goal_size as a threshold
    if goal and np.linalg.norm(np.array(runner.pos) - np.array(goal.pos)) < (runner.radius + goal_size):
        result = "Failure"

    return result

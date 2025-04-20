def check_if_runner_in_sight(agent):
    
    a_s = agent.sensors[0].agent_in_sight
    if a_s != None and a_s.team == "runner":
        return True
    return False


def check_sim_result(world) -> str | None:
    """
    Returns:
    - "Success" if runner is caught by any hunter
    - "Failure" if runner reaches goal
    - None if neither has occurred yet
    """
    import numpy as np
    COLLISION_CLEARANCE = 0.005

    runner = next((agent for agent in world.population if getattr(agent, "team", None) == "runner"), None)
    if runner is None:
        return None

    goal = next((obj for obj in world.objects if isinstance(obj.name, str) and obj.name.lower() == "goal"), None)
    
    if goal is not None:
        goal_radius = getattr(goal, "agent_radius", 0.2)
        if np.linalg.norm(np.array(runner.pos) - np.array(goal.pos)) <= (runner.radius + goal_radius): # distance 
            return "Failure"

    for agent in world.population:
        if getattr(agent, "team", None) != "runner":
            if check_if_runner_in_sight(agent) or np.linalg.norm(np.array(runner.pos) - np.array(agent.pos)) <= (runner.radius + agent.radius + COLLISION_CLEARANCE):
                return "Success"

    return None

import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class NonCVPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="NonCVPolicy")
        self.delta_heading = np.random.uniform(-np.pi/2,np.pi/2)
        self.frequency = np.random.uniform(0.5,5)

    def find_next_action(self, obs, agents, i):
        # Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents.

        heading_increment = self.delta_heading*np.cos(agents[i].step_num/(2*np.pi*self.frequency))
        action = np.array([agents[i].pref_speed, -agents[i].heading_ego_frame+heading_increment])
        return action

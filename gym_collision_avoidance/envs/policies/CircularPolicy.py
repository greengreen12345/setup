import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
import random

class CircularPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="CircularPolicy")

        self.r = np.random.uniform(1, 5)
        self.frequency = np.random.uniform(20, 100)
        self.side =random.choice([-1,1])

    def find_next_action(self, obs, agents, i):
        # Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents.

        """
        radius = self.r
        theta_increment = agents[i].pref_speed/2/np.pi/radius*self.side
        next_x = radius*np.cos(theta+theta_increment)
        next_y = radius * np.sin(theta + theta_increment)

        agents[i].set_state(px=next_x,py=next_y)
        """
        if agents[i].step_num == 0:
            agents[i].heading_global_frame -= np.pi/2

        w = self.side*agents[i].pref_speed/self.r
        theta_increment = w*agents[i].dt_nominal

        action = np.array([agents[i].pref_speed, -theta_increment])
        return action

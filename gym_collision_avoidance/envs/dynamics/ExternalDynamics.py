import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        #self.agent.set_state(px=self.agent.next_state[0],py=self.agent.next_state[1],heading =self.agent.next_state[2])
        return
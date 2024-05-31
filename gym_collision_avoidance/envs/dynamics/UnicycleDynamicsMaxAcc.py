import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math

class UnicycleDynamicsMaxAcc(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.max_turn_rate = 2.0 # rad/s
        self.max_linear_acc = 2.0
        self.max_turn_acc = 3.0
        self.current_speed = 0.0
        self.max_speed = 2.0
        self.current_turning_rate = 0.0
        self.kp = 1.0

    def step(self, action, dt):
        selected_speed = action[0]
        turning_rate = np.clip(action[1]/dt, -self.max_turn_rate, self.max_turn_rate)

        linear_acc =  np.clip(self.kp*(selected_speed - self.current_speed), -self.max_linear_acc, self.max_linear_acc)
        turn_acc = np.clip(self.kp*(turning_rate - self.current_turning_rate), -self.max_turn_acc, self.max_turn_acc)

        self.current_speed += linear_acc*dt
        self.current_speed = np.clip(self.current_speed, -self.max_speed, self.max_speed)
        self.current_turning_rate += turn_acc*dt

        selected_heading = wrap(self.current_turning_rate*dt + self.agent.heading_global_frame)

        dx = self.current_speed * np.cos(selected_heading) * dt
        dy = self.current_speed * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += np.array([dx, dy])

        self.agent.vel_global_frame[0] = self.current_speed * np.cos(selected_heading)
        self.agent.vel_global_frame[1] = self.current_speed * np.sin(selected_heading)
        self.agent.speed_global_frame = self.current_speed
        self.agent.delta_heading_global_frame = wrap(selected_heading -
                                               self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading
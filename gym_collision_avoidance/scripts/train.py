import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
#from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy
import gym_collision_avoidance.envs.test_cases as tc

Config.TRAIN_SINGLE_AGENT = True
Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = 2

test_case_fn = tc.get_testcase_2agents_swap
#test_case_fn = tc.get_testcase_random
policies = {
            'RVO': {
                'policy': RVOPolicy,
                },
            'Learning': {
                 'policy': LearningPolicy,
                 },
            }

num_agents_to_test = [2]
test_case_args = {}


env, one_env = create_env()
agents = []
for i, policy in enumerate(policies):
    one_env.plot_policy_name = policy
    policy_class = policies[policy]['policy']
    test_case_args['agents_policy'] = policy_class
    agent = test_case_fn(i,agents_policy=policy_class)
    #for agent in agents:
    #    if 'checkpt_name' in policies[policy]:
    #        agent.policy.env = env
    agents.append(agent[0])

one_env.set_agents(agents)
one_env.test_case_index = 0
init_obs = env.reset()

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
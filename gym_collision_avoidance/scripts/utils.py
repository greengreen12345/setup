import numpy as np
#from carrl.perturbations import FGSTPerturbation, PSDPerturbation, UniformNoisePerturbation, NoPerturbation, \3
#	plot_loss_in_ball
#from carrl.carrl_network import CARRL, MultiPolicyCARRL
import gym
import os
import pandas as pd
import glob
from stable_baselines.common.vec_env import DummyVecEnv
import inspect
import argparse
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs import test_cases as tc
import yaml
#import gym_cartpole

### These are only here to allow setting of seeds... hmm
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate_epsilon_vector(eps, inds, obs_shape):
	# go from a scalar and indices to a vector in the same shape as the observation
	# e.g. generate_epsilon_vector(0.3, [2,4], (1,5)) ==> np.array([0., 0., 0., 0.3, 0.3])
	np_inds = (np.zeros_like(inds), np.array([inds]))
	eps_vector = np.zeros(obs_shape)
	eps_vector[np_inds] = eps
	return eps_vector


def setup_carrl(gym_to_use, perturber_to_use, model_name, epsilons={}, model_comparison_settings=None):
	### Setup environment using gym.make, plus any custom configs for that env
	if gym_to_use == "CollisionAvoidance-v0":
		# Set config parameters (overriding config.py)
		Config.DT = 0.2
		Config.STATES_IN_OBS = ['dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agent_states']
		Config.ANIMATION_PERIOD_STEPS = 1
		Config.PLOT_CIRCLES_ALONG_TRAJ = True
		Config.GETTING_CLOSE_RANGE = -np.inf
		Config.REWARD_WIGGLY_BEHAVIOR = 0.0
		Config.WIGGLY_BEHAVIOR_THRESHOLD = -np.inf
		Config.RVO_TIME_HORIZON = 1.0
		Config.MAX_TIME_RATIO = 3.

		# Instantiate the environment
		env, one_env = create_env()

		# In case you want to save plots, choose the directory
		one_env.set_plot_save_dir(
			os.path.dirname(os.path.realpath(__file__)) + '/experiments/results/example/')

		# Set agent configuration (start/goal pos, radius, size, policy)
		agents = tc.get_testcase_two_agents_carrl_rvo()
		[agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
		one_env.set_agents(agents)
	elif gym_to_use in ["CartPole-v0", "Pendulum-v0"]:
		if gym_to_use == "CartPole-v0":
			# temp hack trying new cartpole env
			env = DummyVecEnv([lambda: gym.make("CartPole-v0")])
		# env = DummyVecEnv([lambda: gym.make("CartPoleACL-v0")])
		else:
			env = DummyVecEnv([lambda: gym.make(gym_to_use)])
		one_env = env.envs[0].unwrapped
	else:
		raise NotImplementedError

	### Setup CARRL network
	if 'protect' in epsilons:
		# go from a scalar and indices to protect to a vector in the same shape as the observation
		epsilon_robustness = generate_epsilon_vector(epsilons['protect']['scalar_robustness'],
		                                             epsilons['protect']['obs_inds'], (1,) + env.observation_space.shape)
	else:
		epsilon_robustness = None
	model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/" + gym_to_use + "/" + model_name

	if model_comparison_settings is None:
		# The normal case, just load an instance of CARRL with the pickle file
		model = CARRL.load(model_path + ".pkl", epsilon_robustness=epsilon_robustness)
	else:
		# Special version of CARRL that queries multiple policies each step (for analysis in paper)
		model = MultiPolicyCARRL.load(model_path + ".pkl", epsilon_robustness=epsilon_robustness,
		                              settings=model_comparison_settings)

	dqn_obs_ph, dqn_q_values = model.get_obs_and_q_value_phs()

	### Setup Perturber (Adversary/Noise)
	if 'perturb' in epsilons:
		# go from a scalar and indices to perturb to a vector in the same shape as the observation
		epsilon_perturbation = generate_epsilon_vector(epsilons['perturb']['scalar_perturbation'],
		                                               epsilons['perturb']['obs_inds'], (1,) + env.observation_space.shape)
	else:
		epsilon_perturbation = None

	if perturber_to_use == "FGST":
		perturber = FGSTPerturbation(dqn_obs_ph, dqn_q_values, model.graph, model.sess,
		                             epsilon_perturbation=epsilon_perturbation)
	elif perturber_to_use == "PSD":
		perturber = PSDPerturbation(dqn_obs_ph, dqn_q_values, model.graph, model.sess,
		                            epsilon_perturbation=epsilon_perturbation, alpha_vec=epsilon_perturbation / 50.,
		                            num_iterations=100)
	elif perturber_to_use == "UniformNoise":
		perturber = UniformNoisePerturbation(epsilon_perturbation)
	elif perturber_to_use == "none":
		perturber = NoPerturbation()
	else:
		raise NotImplementedError
	return env, one_env, model, perturber


def run_episode(env, one_env, obs, perturber, model, gym_to_use, render=False):
	total_reward = 0
	step = 0
	done = False
	actions = []
	while not done:
		# perturb the observation
		perturbed_obs = perturber.get_perturbed_obs(obs)

		# Query the CARRL agent's policy (returns a robust action)
		action, _ = model.predict(perturbed_obs)
		actions.append(action)

		if type(action) == dict:
			action = action["CARRL"]

		# Adjust the action to be in the shape the particular env requires
		if gym_to_use == "CollisionAvoidance-v0":
			action = [{0: action[0]}]
		elif gym_to_use == "Pendulum-v0":
			action = [action]
		# elif gym_to_use in ["Pendulum-v0", "CartPole-v0"]:
		elif gym_to_use == "CartPole-v0":
			action = action
		else:
			raise NotImplementedError

		# Send some info for collision avoidance env visualization (need a better way to do this)
		# one_env.set_perturbed_info({'perturbed_obs': perturbed_obs[0], 'perturber': perturber})

		# Update the rendering of the environment (optional)
		if render:
			env.render()

		# Take a step in the environment, record reward/steps for logging
		obs, rewards, done, which_agents_done = env.step(action)
		total_reward += rewards[0]
		step += 1

	# After end of episode, store some statistics about the environment
	# Some stats apply to every gym env...
	generic_episode_stats = {
		'total_reward': total_reward,
		'steps': step,
		'actions': actions
	}

	# Other stats are specific to one gym env...
	if gym_to_use == "CollisionAvoidance-v0":
		agents = one_env.agents
		# agents = one_env.prev_episode_agents
		time_to_goal = np.array([a.t for a in agents])
		extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])
		collision = np.array(
			np.any([a.in_collision for a in agents])).tolist()
		all_at_goal = np.array(
			np.all([a.is_at_goal for a in agents])).tolist()
		any_stuck = np.array(
			np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
		outcome = "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
		specific_episode_stats = {
			'num_agents': len(agents),
			'time_to_goal': time_to_goal,
			'total_time_to_goal': np.sum(time_to_goal),
			'extra_time_to_goal': extra_time_to_goal,
			'collision': collision,
			'all_at_goal': all_at_goal,
			'any_stuck': any_stuck,
			'outcome': outcome,
			'ego_agent_policy': agents[0].policy.str,
			'other_agent_policy': agents[1].policy.str
		}
	elif gym_to_use == "CartPole-v0":
		specific_episode_stats = {}
	elif gym_to_use == "Pendulum-v0":
		specific_episode_stats = {}
	else:
		raise NotImplementedError

	# Merge all stats into a single dict
	episode_stats = {**generic_episode_stats, **specific_episode_stats}

	env.reset()

	return episode_stats


def store_stats(df, hyperparameters, episode_stats):
	# Add a new row to the pandas DataFrame (a table of results, where each row is an episode)
	# that contains the hyperparams and stats from that episode, for logging purposes
	df_columns = {**hyperparameters, **episode_stats}
	df = df.append(df_columns, ignore_index=True)
	return df


def reset_env(env, gym_to_use, seed=None, test_case_fn=None, test_case_args=None, test_case=None, one_env=None,
              perturbation_str=None, epsilon_scalar_robustness=None, results_dir=None):
	# Do any env-specific bookkeeping before resetting here, to keep it out of high-level scripts
	seed = 0 if seed is None else seed
	set_seeds(seed)
	if gym_to_use == "CollisionAvoidance-v0":
		if test_case_fn is not None:
			test_case_args['test_case_index'] = test_case
			test_case_args['seed'] = seed

			# Before running test_case_fn, make sure we didn't provide any args it doesn't accept
			test_case_fn_args = inspect.getargspec(test_case_fn).args
			test_case_args_keys = list(test_case_args.keys())
			for key in test_case_args_keys:
				if key not in test_case_fn_args:
					# print("{} doesn't accept {} -- removing".format(test_case_fn, key))
					del test_case_args[key]
			agents = test_case_fn(**test_case_args)

			[agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
			one_env.set_agents(agents)
			one_env.test_case_index = test_case
		if Config.SAVE_EPISODE_PLOTS and results_dir is not None:
			plot_save_dir = results_dir + '/trajs/'
			if perturbation_str is not None:
				plot_save_dir = plot_save_dir + '{}/'.format(perturbation_str)
			if epsilon_scalar_robustness is not None:
				plot_save_dir = plot_save_dir + 'rob_{}/'.format(round(epsilon_scalar_robustness, 4))
			if seed is not None:
				plot_save_dir = plot_save_dir + 'seed_{}/'.format(str(seed).zfill(3))
			one_env.set_plot_save_dir(plot_save_dir)
	elif gym_to_use in ["CartPole-v0", "Pendulum-v0"]:
		one_env.seed(seed)
	obs = env.reset()
	return obs


def set_seeds(seed=0):
	# Set the seed for every library that might use random number generators
	np.random.seed(seed)
	tf.compat.v1.set_random_seed(seed)

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = "".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)

def get_wrapper_class(hyperparams):
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.DoneOnSuccessWrapper:
            reward_offset: 1.0
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams: (dict)
    :return: a subclass of gym.Wrapper (class object) you can use to
             create another Gym env giving an original env.
    """

    def get_module_name(wrapper_name):
        return '.'.join(wrapper_name.split('.')[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split('.')[-1]

    if 'env_wrapper' in hyperparams.keys():
        wrapper_name = hyperparams.get('env_wrapper')

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env):
            """
            :param env: (gym.Env)
            :return: (gym.Env)
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env
        return wrap_env
    else:
        return None

def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
                if test_mode:
                    normalize_kwargs['norm_reward'] = norm_reward
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path


def find_saved_model(algo, log_path, env_id, load_best=False):
    """
    :param algo: (str)
    :param log_path: (str) Path to the directory with the saved model
    :param env_id: (str)
    :param load_best: (bool)
    :return: (str) Path to the saved model
    """
    model_path, found = None, False
    for ext in ['pkl', 'zip']:
        model_path = "{}/{}.{}".format(log_path, env_id, ext)
        found = os.path.isfile(model_path)
        if found:
            break

    if load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError("No model found for {} on {}, path: {}".format(algo, env_id, model_path))
    return model_path
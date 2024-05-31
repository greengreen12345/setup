import numpy as np

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    COLLISION_AVOIDANCE = True
    continuous, discrete = range(2) # Initialize game types as enum
    ACTION_SPACE_TYPE   = continuous

    ANIMATE_EPISODES    = True
    SHOW_EPISODE_PLOTS = True
    SAVE_EPISODE_PLOTS = False
    TRAIN_MODE           = False # Enable to see the trained agent in action (for testing)
    PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
    EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)
    TRAIN_SINGLE_AGENT = True

    LSTM_HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    NUM_HIDDEN_UNITS = 64
    NETWORK = "mfe_network"
    GAMMA = 0.99
    LEARNING_RATE = 1e-3

    #########################################################################
    # COLLISION AVOIDANCE PARAMETER
    NUM_TEST_CASES = 50
    PLOT_EVERY_N_EPISODES = 10000# for tensorboard visualization
    DT             = 0.2 # seconds between simulation time steps
    REWARD_AT_GOAL = 3.0 # reward given when agent reaches goal position
    REWARD_COLLISION_WITH_AGENT = -10.0 # reward given when agent collides with another agent
    REWARD_TIMEOUT = -10.0 # reward given for not reaching the goal
    REWARD_INFEASIBLE = 0.0
    REWARD_COLLISION_WITH_WALL = -0.25 # reward given when agent collides with wall
    REWARD_GETTING_CLOSE   = 0.0 # reward when agent gets close to another agent (unused?)
    REWARD_ENTERED_NORM_ZONE   = 0.0 # reward when agent enters another agent's social zone
    REWARD_TIME_STEP   = -0.01 # default reward given if none of the others apply (encourage speed)
    REWARD_DISTANCE_TO_GOAL = 0.0  # default reward given if none of the others apply (encourage speed)
    REWARD_WIGGLY_BEHAVIOR = 0.0
    WIGGLY_BEHAVIOR_THRESHOLD = 0.0
    ENABLE_COLLISION_AVOIDANCE = True
    COLLISION_DIST = 0.5 # meters between agents' boundaries for collision
    GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
    JOINT_MPC_RL_TRAINING = False # select the action that has highets reward (mpc/rl)
    CURRICULUM_LEARNING = True
    HOMOGENEOUS_TESTING = True

    LASERSCAN_LENGTH = 512 # num range readings in one scan
    NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
    NUM_PAST_ACTIONS_IN_STATE = 0

    NEAR_GOAL_THRESHOLD = 0.75
    MAX_TIME_RATIO = 3.0 # agent has this number times the straight-line-time to reach its goal before "timing out"

    SENSING_HORIZON  = np.inf
    # SENSING_HORIZON  = 3.0

    RVO_TIME_HORIZON = 5.0
    RVO_COLLAB_COEFF = 0.5
    RVO_ANTI_COLLAB_T = 1.0

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
    MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
    MAX_NUM_OTHER_AGENTS_OBSERVED = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

    PLOT_CIRCLES_ALONG_TRAJ = True
    ANIMATION_PERIOD_STEPS = 4 # plot every n-th DT step (if animate mode on)
    PLT_LIMITS = None
    PLT_FIG_SIZE = (10, 8)

    STATES_IN_OBS = ['dist_to_goal', 'rel_goal','radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states']
    # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo', 'laserscan']
    # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo'] # 2-agent net
    # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM
    STATES_NOT_USED_IN_POLICY = ['use_ppo', 'num_other_agents']
    STATE_INFO_DICT = {
        'dist_to_goal': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [-np.inf, np.inf],
            'attr': 'get_agent_data("dist_to_goal")',
            'std': np.array([5.], dtype=np.float32),
            'mean': np.array([0.], dtype=np.float32)
            },
        'radius': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [0, np.inf],
            'attr': 'get_agent_data("radius")',
            'std': np.array([1.0], dtype=np.float32),
            'mean': np.array([0.5], dtype=np.float32)
            },
        'rel_goal': {
            'dtype': np.float32,
            'size': 2,
            'bounds': [-np.inf, np.inf],
            'attr': 'get_agent_data("rel_goal")',
            'std': np.array([10.0], dtype=np.float32),
            'mean': np.array([0.], dtype=np.float32)
            },
        'heading_ego_frame': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [-np.pi, np.pi],
            'attr': 'get_agent_data("heading_ego_frame")',
            'std': np.array([3.14], dtype=np.float32),
            'mean': np.array([0.], dtype=np.float32)
        },
        'pref_speed': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [0, np.inf],
            'attr': 'get_agent_data("pref_speed")',
            'std': np.array([1.0], dtype=np.float32),
            'mean': np.array([1.0], dtype=np.float32)
            },
        'num_other_agents': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [0, np.inf],
            'attr': 'get_agent_data("num_other_agents_observed")',
            'std': np.array([1.0], dtype=np.float32),
            'mean': np.array([1.0], dtype=np.float32)
            },
        'other_agent_states': {
            'dtype': np.float32,
            'size': 9,
            'bounds': [-np.inf, np.inf],
            'attr': 'get_agent_data("other_agent_states")',
            'std': np.array([5.0, 5.0,5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
            'mean': np.array([0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
            },
        'other_agents_states': {
            'dtype': np.float32,
            'size': (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT,9),
            'bounds': [-np.inf, np.inf],
            'attr': 'get_sensor_data("other_agents_states")',
            'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0, 5.0, 1.0], dtype=np.float32), (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1)),
            'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0], dtype=np.float32), (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1)),
            },
        'laserscan': {
            'dtype': np.float32,
            'size': LASERSCAN_LENGTH,
            'bounds': [0., 6.],
            'attr': 'get_sensor_data("laserscan")',
            'std': 5.*np.ones((LASERSCAN_LENGTH), dtype=np.float32),
            'mean': 5.*np.ones((LASERSCAN_LENGTH), dtype=np.float32)
            },
        'use_ppo': {
            'dtype': np.float32,
            'size': 1,
            'bounds': [0., 1.],
            'attr': 'get_agent_data_equiv("policy.str", "learning")'
            }
        }
    MEAN_OBS = {}; STD_OBS = {}
    for state in STATES_IN_OBS:
        if 'mean' in STATE_INFO_DICT[state]:
            MEAN_OBS[state] = STATE_INFO_DICT[state]['mean']
        if 'std' in STATE_INFO_DICT[state]:
            STD_OBS[state] = STATE_INFO_DICT[state]['std']

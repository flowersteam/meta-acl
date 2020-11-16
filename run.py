import argparse
from teachDRL.spinup.utils.run_utils import setup_logger_kwargs
from teachDRL.spinup.algos.sac.sac import sac
from teachDRL.spinup.algos.sac import core
import gym
import teachDRL.gym_flowers
from teachDRL.teachers.teacher_controller import TeacherController
from collections import OrderedDict
import os
import numpy as np

from teachDRL.teachers.algos.random_replay_buffer import ReplayBuffer
import pickle
import os
import json

def load_human_expert_trajectory(folder_path, human_alp=None):
    # 1 - loading the trajectory
    try:
        gmms_means, gmms_covs, gmms_mean_rew = pickle.load(open(os.path.join(folder_path, 'human_made_curriculum.pkl'), "rb"))
    except:
        print('Unable to load expert trajectory data: {}'.format(folder_path))
        exit(1)
    gmms_mean_rew = np.interp(gmms_mean_rew, (-150, 350), (0, 1))
    if human_alp is not None: #change alp
        print(gmms_means)
        for gmm_means in gmms_means:
            for gaussian_means in gmm_means:
                gaussian_means[-1] = human_alp/100.0
        print('after')
        print(gmms_means)
    return gmms_means, gmms_covs, gmms_mean_rew

def load_expert_trajectory(folder_path, alp_thr=0.1, max_steps=10000000):
    print('loading {} data'.format(folder_path))
    # 1 - loading the trajectory
    data = {}
    # select seeded run
    try:
        env_params_dict = pickle.load(open(os.path.join(folder_path, 'env_params_save.pkl'), "rb"))
    except:
        print('Unable to load expert trajectory data: {}'.format(folder_path))
        exit(1)
    for k, v in env_params_dict.items():
        if k == 'means' or k == 'covariances' or k == 'weights' or k == 'env_train_len' or k == 'episodes' \
           or k == 'env_train_rewards':
            data[k] = v
        if k == 'egt_means' or k == 'egt_covariances' or k == 'egt_weights' or k == 'egt_env_train_len' or k == 'egt_episodes' \
                or k == 'egt_env_train_rewards':
            data[k[4:]] = v

    # 2 - pre-processing expert trajectory
    # removing low-alp gaussians
    processed_gmms_means = []
    processed_gmms_covs = []
    processed_gmms_mean_rew = []
    idx_removed_gmms = []
    step_nb = 0
    gmm_step = data['episodes'][0]
    for i, (gmm_means, gmm_covs, episode) in enumerate(zip(data["means"], data["covariances"], data['episodes'])):
        step_nb += sum(data['env_train_len'][i*gmm_step:(i+1)*gmm_step])
        processed_gmm_means = []
        processed_gmm_covs = []
        all_rewards = np.interp(data['env_train_rewards'][episode:episode + gmm_step], (-150, 350), (0, 1))  # from gmm
        rewards = all_rewards[-50:]  # consider mean reward after some training on the GMM
        mean_reward = np.mean(rewards)
        for j, (means, covs) in enumerate(zip(gmm_means, gmm_covs)):
            if means[-1] > alp_thr:  # last mean is ALP dimension
                # add gaussian
                processed_gmm_means.append(means)
                processed_gmm_covs.append(covs)
        if not processed_gmm_means == []:  # gmm not empty after pre-process, lets add it
            processed_gmms_means.append(processed_gmm_means)
            processed_gmms_covs.append(processed_gmm_covs)
            processed_gmms_mean_rew.append(mean_reward)
        else:
            idx_removed_gmms.append(i)
        if step_nb > max_steps:
            break
    print('idx of removed gmms ({}/{}) in expert traj: {}'.format(len(data['means']) - len(processed_gmms_means),
                                                                  len(data['means']),
                                                                  idx_removed_gmms))
    print('number of steps: {}'.format(step_nb))
    return processed_gmms_means, processed_gmms_covs, processed_gmms_mean_rew


# Argument definition
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--seed', '-s', type=int, default=0)

# Deep RL student arguments, so far only works with SAC
parser.add_argument('--hid', type=str, default=-1)  # number of neurons in hidden layers
parser.add_argument('--l', type=int, default=1)  # number of hidden layers
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--nb_env_steps', type=float, default=10.0)  # Training time, expressed in Millions of env steps
parser.add_argument('--gpu_id', type=int, default=-1)  # default is no GPU
parser.add_argument('--ent_coef', type=float, default=0.005)
parser.add_argument('--max_ep_len', type=int, default=2000)
parser.add_argument('--steps_per_ep', type=int, default=500000)  # nb env steps/epoch (stay above max_ep_len and nb_env_steps)
parser.add_argument('--buf_size', type=int, default=2000000)
parser.add_argument('--nb_test_episodes', type=int, default=225)
parser.add_argument('--custom_test_set', '-cts', action='store_true')  # predefined test sets for walker
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--train_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--half_save', '-hs', action='store_true')
parser.add_argument('--random_net', '-rndnn', action='store_true')
parser.add_argument('--random_leg_s', '-rndls', action='store_true')

# Parameterized bipedal walker arguments, so far only works with bipedal-walker-continuous-v0
parser.add_argument('--env', type=str, default="wc-env-v0")#classroom-bipedal-walker-continuous-v0
parser.add_argument('--no_short_walker_help', '-nswh', action='store_true')
parser.add_argument('--fat_short_walker', '-fsw', action='store_true')
parser.add_argument('--min_torque', '-mint', type=int, default=20)

# Choose student (walker morphology)
parser.add_argument('--leg_size', '-leg_s', type=float, default=None)  # leg size normalized between 0 and 1
parser.add_argument('--walker_type', '-walk_t', type=float, default=None)  # if in [0,0.5], bipedal and if in [0.5,1], quadru

# Selection of parameter space for wc env
parser.add_argument('--max_obstacle_spacing', type=float, default=6)
parser.add_argument('--min_gap_pos', type=float, default=2.5)
parser.add_argument('--max_gap_pos', type=float, default=7.5)
parser.add_argument('--climbing_surface_size', type=float, default=0.5)
parser.add_argument('--agent_type', type=str, default=None)

# Teacher-specific arguments:
parser.add_argument('--teacher', type=str, default="ALP-GMM")  # ALP-GMM, Covar-GMM, RIAC, Oracle, Random, ADR, AGAIN
parser.add_argument('--random_percentage', '-rnd', type=float, default=None)

# ALPGMM (Absolute Learning Progress - Gaussian Mixture Model) related arguments
parser.add_argument('--gmm_fitness_fun', '-fit', type=str, default=None)
parser.add_argument('--nb_em_init', type=int, default=None)
parser.add_argument('--min_k', type=int, default=None)
parser.add_argument('--max_k', type=int, default=None)
parser.add_argument('--fit_rate', type=int, default=None)
parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
parser.add_argument('--alp_max_size', type=float, default=None)  # alp-computer window, expressed in Millions of env steps

# CovarGMM related arguments
parser.add_argument('--absolute_lp', '-alp', action='store_true')

# RIAC related arguments
parser.add_argument('--max_region_size', type=int, default=None)
parser.add_argument('--alp_window_size', type=int, default=None)

# EGT related arguments
parser.add_argument('--use_alpgmm', '-alpgmm', action='store_true')
parser.add_argument('--use_human', '-human', type=float, default=None)  # ALP value of human ALP in expert traj
parser.add_argument('--r_list_len', '-rlen', type=int, default=10)
parser.add_argument('--expert_type', type=str, default="R")
parser.add_argument('--expert_weights', '-expw', action='store_true')
#parser.add_argument('--stop_R', '-sR', action='store_true',default=True)
parser.add_argument('--tol_ratio', '-tolr', type=int, default=100)
parser.add_argument('--prior_run_name', '-prn', type=str, default="15-04_ALP-GMM_classroom")
parser.add_argument('--walker_mutation', '-w_mut', type=str, default=None)

# AGAIN related arguments
parser.add_argument('--k', type=int, default=3)  # nb considered closest previously trained students
parser.add_argument('--pretrain_epochs', '-pt', type=int, default=1)
parser.add_argument('--restart_after_pretrain', '-rap', action='store_true')
parser.add_argument('--use_ground_truth', '-gt', action='store_true')
parser.add_argument('--random_expert', '-re', action='store_true')
parser.add_argument('--in_end_rnd', '-ier', type=float, default=None)

# ADR related arguments
parser.add_argument('--boundary_sampling_p', '-bsp', type=float, default=0.5)
parser.add_argument('--step_size', '-ss', type=float, default=0.05)
parser.add_argument('--reward_thr', '-rt', type=int, default=230)
parser.add_argument('--queue_len', '-ql', type=int, default=10)
parser.add_argument('--classroom_filename', '-cf', type=str, default='student_history')

args = parser.parse_args()
assert args.seed is not None
np.random.seed(args.seed)
# Bind this run to specific GPU if there is one
if args.gpu_id != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Set up Student's DeepNN architecture if provided
ac_kwargs = dict()
if args.hid != -1:
    ac_kwargs['hidden_sizes'] = [int(nb) for nb in args.hid.split('x')]

if args.random_net: # vary network sizes, so far either --> in meta-acl paper only [400,300] was used
    possible_nets = [[400,300], [200,150], [125,125]]
    ac_kwargs['hidden_sizes'] = possible_nets[np.random.randint(len(possible_nets))]
else:
    if args.hid == -1:
        ac_kwargs['hidden_sizes'] = [400,300]

pretrained_model = None
start_steps = 10000
half_save = False
if args.half_save is True:
    half_save = True

# Set bounds for environment's parameter space format:[min, max, nb_dimensions] (if no nb_dimensions, assumes only 1)
param_env_bounds = OrderedDict()
# wc env
if args.min_gap_pos and args.max_gap_pos and args.env == 'wc-env-v0':
    param_env_bounds['gap_pos'] = [args.min_gap_pos, args.max_gap_pos]
    param_env_bounds['obstacle_spacing'] = [0, args.max_obstacle_spacing]

if args.max_obstacle_spacing is not None:
    param_env_bounds['obstacle_spacing'] = [0, args.max_obstacle_spacing]

# Set Teacher hyperparameters
params = {}
if args.teacher == 'ALP-GMM':
    if args.gmm_fitness_fun is not None:
        params['gmm_fitness_fun'] = args.gmm_fitness_fun
    if args.min_k is not None and args.max_k is not None:
        params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
    if args.weighted_gmm is True:
        params['weighted_gmm'] = args.weighted_gmm
    if args.nb_em_init is not None:
        params['nb_em_init'] = args.nb_em_init
    if args.fit_rate is not None:
        params['fit_rate'] = args.fit_rate
    if args.alp_max_size is not None:
        params['alp_max_size'] = int(args.alp_max_size*1e6)
    if args.random_percentage is not None:
        params["random_task_ratio"] = args.random_percentage / 100.0

elif args.teacher == 'Covar-GMM':
    if args.absolute_lp is True:
        params['absolute_lp'] = args.absolute_lp
elif args.teacher == "RIAC":
    if args.max_region_size is not None:
        params['max_region_size'] = args.max_region_size
    if args.alp_window_size is not None:
        params['alp_window_size'] = args.alp_window_size
elif args.teacher == "Oracle":
    if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
        params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
    elif 'poly_shape' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 12
        print('hih')
    elif 'stump_seq' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 10
    else:
        print('Oracle not defined for this parameter space')
        exit(1)
elif args.teacher == "EGT":
    # load expert traj
    # function to load and extract expert trajectory from folder
    folder_path = None
    if args.use_human is not None: # use human made expert curriculum
        folder_path = 'teachDRL/data/elders_knowledge/'
        params['expert_gmms'] = load_human_expert_trajectory(folder_path, human_alp=args.use_human)
        print('Using human expert !')
    elif args.leg_size == "default":
        folder_path = 'teachDRL/data/elders_knowledge/ALP-GMMcdefaultfinalexpert10rnd04-02/ALP-GMMcdefaultfinalexpert10rnd04-02_s'
        seed_str = str(args.seed)
        folder_path = folder_path + seed_str + '/'
        params['expert_gmms'] = load_expert_trajectory(folder_path)
    elif args.leg_size == "short":
        folder_path = 'teachDRL/data/elders_knowledge/ALP-GMMcshortfinalexpert10rnd05-02/ALP-GMMcshortfinalexpert10rnd05-02_s'
        seed_str = str(args.seed)
        folder_path = folder_path + seed_str + '/'
        params['expert_gmms'] = load_expert_trajectory(folder_path)
    elif args.leg_size is None:  # Use classroom for EGT
        folder_path = 'teachDRL/data/elders_knowledge/{}/{}_s'.format(args.prior_run_name, args.prior_run_name)
        seed_str = str(args.seed)
        folder_path = folder_path + seed_str + '/'
        params['expert_gmms'] = load_expert_trajectory(folder_path)
        # get leg_type
        config_path = open(os.path.join(folder_path, 'config.json'))
        config = json.load(config_path)
        if type(config['env_init']) == type([]):
            env_init = config['env_init']
        else:
            env_init_str = config['env_init'][1:-1]
            env_init = np.fromstring(env_init_str, sep=' ')
        print('oldenvinit {}'.format(env_init))
        args.leg_size = env_init[0]
        args.walker_type = env_init[1]

        if args.walker_mutation is not None:
            if args.walker_mutation == 'seed':
                # change seed (which will change networks weights wrt to loaded expert)
                args.seed += 1000
            elif args.walker_mutation == 'leg_s':
                # slightly mutate leg size
                args.leg_size += np.random.normal(0, 0.1)
                args.leg_size = np.clip(args.leg_size, 0, 1)
            else:
                print('unknown')
                exit(0)
        print('mutated leg_size: {}'.format(args.leg_size))


    if args.expert_weights:  # re-use expert weights
        # load previous policy
        #sess, model = load_policy(folder_path)
        checkpoint_path = folder_path + 'simple_save20/variables/variables'
        print("loading expert policy: {}".format(checkpoint_path))
        pretrained_model = checkpoint_path
        start_steps = 0  # no initial random steps when starting from trained weights
    if args.use_alpgmm:  # always use stopR
        print('USING STOPR !!')
        exit(0)
        params['stop_R'] = True

    params['use_alpgmm'] = args.use_alpgmm
    params['expert_type'] = args.expert_type
    params['expert_name'] = folder_path
    params['r_list_len'] = args.r_list_len
    params['tol_ratio'] = args.tol_ratio / 100.0
    if args.random_percentage is not None:
        params["random_task_ratio"] = args.random_percentage / 100.0

elif args.teacher == "AGAIN":
    # load expert traj
    # function to load and extract expert trajectory from folder
    params['stop_R'] = False
    if args.use_alpgmm:  # always use stopR with AGAIN
        params['stop_R'] = True

    params['use_alpgmm'] = args.use_alpgmm
    params['expert_type'] = args.expert_type
    params['r_list_len'] = args.r_list_len
    params['tol_ratio'] = args.tol_ratio / 100.0
    params['k'] = args.k
    params['pretrain_epochs'] = args.pretrain_epochs
    params['restart_after_pretrain'] = True if args.restart_after_pretrain else False
    params['use_ground_truth'] = True if args.use_ground_truth else False
    params['classroom_filename'] = args.classroom_filename
    params['random_expert'] = True if args.random_expert else False
    if args.in_end_rnd is not None:
        params['in_end_rnd'] = args.in_end_rnd / 100.0

    if args.random_percentage is not None:
        params["random_task_ratio"] = args.random_percentage / 100.0

elif args.teacher == "ADR":
    params['step_size'] = [args.step_size, args.step_size]  # Warning, this trick only works on 2d Parkour env
    params['boundary_sampling_p'] = args.boundary_sampling_p
    params['reward_thr'] = args.reward_thr
    params['queue_len'] = args.queue_len
    params['initial_task'] = [(np.random.random()*args.max_gap_pos)+args.min_gap_pos, (np.random.random()*args.max_obstacle_spacing)]  # Warning, this trick only works on 2d Parkour env
    params['initial_task'] = np.round(params['initial_task'],2)
    print('ADR initial task is {}'.format(params['initial_task']))
# parser.add_argument('--min_gap_pos', type=float, default=2.5)
# parser.add_argument('--max_gap_pos', type=float, default=7.5)
#env_f = lambda: gym.make(args.env)

# Extract walker params
env_init_dict = {}
env_f = None
env_init = None
if args.env == "classroom-bipedal-walker-continuous-v0":
    walker_params = None
    if args.leg_size is not None and args.walker_type is not None:
        walker_params = [args.leg_size, args.walker_type]
    if args.no_short_walker_help is True:
        env_init_dict['help_short_walker'] = False
        env_init_dict['min_torque'] = args.min_torque
    else:
        env_init_dict['help_short_walker'] = True
    if args.fat_short_walker is True:
        env_init_dict['fat_short_walker'] = True
    else:
        env_init_dict['fat_short_walker'] = False
    print(walker_params)
    env_init_dict['params'] = walker_params

    env = gym.make(args.env)
    env_params = env.env.my_init(env_init_dict)
    print('env is:{}'.format(env_params))
    def run_env(env, env_param_dict):
        env = gym.make(env)
        e_i = env.env.my_init(env_param_dict)
        print('other env is {}'.format(env_params))
        return env
    env_f = lambda env_i: run_env(args.env, env_i)
    env_init_dict['params'] = env_params
else:
    env_init_dict = {'water_level': 0}
    if args.agent_type is None:  # random agent type assignment
        if np.random.random() > 0.5:
            env_init_dict['agent_body_type'] = 'old_classic_bipedal'
        else:
            env_init_dict['agent_body_type'] = 'climbing_chest_profile_chimpanzee'
    else:
        if args.agent_type == 'climber':
            env_init_dict['agent_body_type'] = 'climbing_chest_profile_chimpanzee'
        elif args.agent_type == 'walker':
            env_init_dict['agent_body_type'] = 'old_classic_bipedal'
        else:
            print('unknown agent type')
            exit(1)
    env_params = [0] if env_init_dict['agent_body_type'] == 'old_classic_bipedal' else [1]
    leg_s = 1
    if args.random_leg_s:
        leg_s = np.round((np.random.random() / 2) + 0.5, 2)  # leg_s between random 0.5 and 1.0
    elif args.leg_size is not None:
        leg_s = args.leg_size
    env_init_dict['leg_s'] = leg_s
    env_f = lambda env_i: gym.make(args.env, **env_i)

env = env_f(env_init_dict)
#env_init = env.env.my_init(env_init_dict)
if 'agent_body_type' in env_init_dict:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
else:
    obs_dim = env.env.observation_space.shape[0]
    act_dim = env.env.action_space.shape[0]
env.close()


params['student_params'] = env_params
params['student_params'].append(leg_s)
#add network size string
nn = ac_kwargs['hidden_sizes'][0]
params['student_params'].append(nn)
# Custom walker test set for wc env
custom_test_param_vec = None
args.nb_custom_tests = 0
if args.custom_test_set and env_init_dict['agent_body_type'] == 'old_classic_bipedal':
    nb_steps = 10
    d1 = np.linspace(2.5, 3.6, nb_steps, endpoint=True)
    d2 = np.linspace(0, 6, nb_steps, endpoint=True)
    custom_test_param_vec = np.transpose([np.tile(d1, len(d2)), np.repeat(d2, len(d1))])  # cartesian product
    args.nb_custom_tests = 100

# Initialize teacher
Teacher = TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds,
                            seed=args.seed, teacher_params=params, custom_test_param_vec=custom_test_param_vec)

replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.buf_size)

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
# Launch Student training
nb_test_episodes = args.nb_test_episodes + args.nb_custom_tests
sac(env_f, actor_critic=core.mlp_actor_critic, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed,
    epochs=int((args.nb_env_steps*1e6)//args.steps_per_ep), start_steps=start_steps,
    logger_kwargs=logger_kwargs, alpha=args.ent_coef, max_ep_len=args.max_ep_len, steps_per_epoch=args.steps_per_ep,
    replay_buffer=replay_buffer, env_init=env_init_dict, env_name=args.env, nb_test_episodes=nb_test_episodes, lr=args.lr,
    train_freq=args.train_freq, batch_size=args.batch_size, Teacher=Teacher, run_args=args, half_save=half_save, pretrained_model=pretrained_model)
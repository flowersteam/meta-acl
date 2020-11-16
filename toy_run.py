import argparse
from teachDRL.spinup.utils.run_utils import setup_logger_kwargs
from teachDRL.spinup.utils.logx import EpochLogger
from teachDRL.teachers.teacher_controller import TeacherController
from teachDRL.teachers.algos.alp_gmm import ALPGMM
#from teachDRL.teachers.algos.egt import EGT
from teachDRL.teachers.algos.again import AGAIN
from teachDRL.teachers.algos.adr import ADR
from teachDRL.toy_env.classroom_toy_env import ClassroomToyEnv
from teachDRL.toy_env.classroom_toy_env import ClassroomToyEnvV2
from teachDRL.teachers.algos.random_teacher import RandomTeacher
from collections import OrderedDict
import os
import numpy as np
import pickle
import os
import json
import time
import copy

def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if len(bounds) == 2:
            param_dict[name] = param[i]
            cpt += 1
        elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
            nb_dims = bounds[2]
            param_dict[name] = param[i:i+nb_dims]
            cpt += nb_dims
    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        #print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)


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
parser.add_argument('--nb_episodes', type=float, default=2)  # Training time, expressed in 100K episodes
parser.add_argument('--epoch_step', type=float, default=0.1)  # Test interval, expressed in 100K episodes

parser.add_argument('--nb_cubes', type=int, default=20)
parser.add_argument('--nb_rot', type=int, default=None)
parser.add_argument('--toy_env_2', '-v2', action='store_true')
parser.add_argument('--rnd_start_cube', '-rsc', action='store_true')
parser.add_argument('--light', action='store_true')  # lightweight saves
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
parser.add_argument('--stop_R', '-sR', action='store_true')
parser.add_argument('--tol_ratio', '-tolr', type=int, default=100)
parser.add_argument('--prior_run_name', '-prn', type=str, default="15-04_ALP-GMM_classroom")
parser.add_argument('--walker_mutation', '-w_mut', type=str, default=None)

# AGAIN related arguments
parser.add_argument('--k', type=int, default=5)  # nb considered closest previously trained students
parser.add_argument('--pretrain_epochs', '-pt', type=int, default=1)
parser.add_argument('--restart_after_pretrain', '-rap', action='store_true')
parser.add_argument('--use_ground_truth', '-gt', action='store_true')
parser.add_argument('--random_expert', '-re', action='store_true')


# ADR related arguments
parser.add_argument('--boundary_sampling_p', '-bsp', type=float, default=0.5)
parser.add_argument('--step_size', '-ss', type=float, default=0.05)
parser.add_argument('--reward_thr', '-rt', type=int, default=1)
parser.add_argument('--queue_len', '-ql', type=int, default=10)

parser.add_argument('--classroom_filename', '-cf', type=str, default=None)
parser.add_argument('--classroom_portion', '-cp', type=int, default=100)  # percentage of classroom data that is used

# LP replay buffer
parser.add_argument('--replay', type=str, default="default")  # replay buffer type ("default", "lp","alp","rew","per")
parser.add_argument('--replay_alpha', type=float, default=1.0)


#################################################
##############  Parse arguments   ###############
#################################################
args = parser.parse_args()

# seed run
seed = args.seed
assert seed is not None
np.random.seed(seed)

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
# elif args.teacher == "Oracle":
#     if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
#         params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
#     elif 'poly_shape' in param_env_bounds:
#         params['window_step_vector'] = [0.1] * 12
#         print('hih')
#     elif 'stump_seq' in param_env_bounds:
#         params['window_step_vector'] = [0.1] * 10
#     else:
#         print('Oracle not defined for this parameter space')
#         exit(1)
elif args.teacher == "EGT":
    # load expert traj
    # function to load and extract expert trajectory from folder
    folder_path = None

    if args.expert_weights:  # re-use expert weights
        # load previous policy
        #sess, model = load_policy(folder_path)
        checkpoint_path = folder_path + 'simple_save20/variables/variables'
        print("loading expert policy: {}".format(checkpoint_path))
        pretrained_model = checkpoint_path
        start_steps = 0  # no initial random steps when starting from trained weights
    if args.stop_R:
        params['stop_R'] = args.stop_R

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

    if args.stop_R:
        params['stop_R'] = args.stop_R

    params['use_alpgmm'] = args.use_alpgmm
    params['expert_type'] = args.expert_type
    params['r_list_len'] = args.r_list_len
    params['tol_ratio'] = args.tol_ratio / 100.0
    params['k'] = args.k
    params['pretrain_epochs'] = args.pretrain_epochs
    params['restart_after_pretrain'] = True if args.restart_after_pretrain else False
    params['use_ground_truth'] = True if args.use_ground_truth else False
    params['classroom_filename'] = args.classroom_filename #'toy_env_v2_student_history' if args.toy_env_2 else
    params['random_expert'] = True if args.random_expert else False
    params['classroom_portion'] = args.classroom_portion

    if args.random_percentage is not None:
        params["random_task_ratio"] = args.random_percentage / 100.0

elif args.teacher == "ADR":
    params['step_size'] = [args.step_size, args.step_size*2]  # Warning, this trick only works on bipedal walker env
    params['boundary_sampling_p'] = args.boundary_sampling_p
    params['reward_thr'] = args.reward_thr
    params['queue_len'] = args.queue_len
    params['initial_task'] = [0,0]
env_f = lambda: gym.make(args.env)

env_init_dict = {}

#env_init_dict['params'] = walker_params

# if args.use_ground_truth:
#     params['student_params'] = env_init
# env_init_dict['params'] = env_init

# INIT logger
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
logger = EpochLogger(**logger_kwargs)
save_dir = logger.output_dir

#################################################
############## Initialize teacher ###############
#################################################

#SEED EXPERIMENT
np.random.seed(args.seed)

Teacher = None  #TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds,
                #             seed=args.seed, teacher_params=params)
teacher_name = args.teacher
teacher_params = params
mins = [0,0]
maxs = [1,1]

# Initialize toy env
nb_dims = 2


if args.nb_cubes is None:
    nb_cubes = np.random.randint(10,31)
else:
    nb_cubes = args.nb_cubes
if args.rnd_start_cube:
    start_cube_idx = np.random.randint(0,400)
else:
    start_cube_idx = args.seed % (nb_cubes**nb_dims)
print("start cube idx: {}".format(start_cube_idx))
if args.nb_rot is None:
    nb_task_space_rot = np.random.randint(0,4)
else:
    nb_task_space_rot = args.nb_rot

if args.toy_env_2:
    teacher_params['student_params'] = [start_cube_idx]
else:
    teacher_params['student_params'] = [nb_task_space_rot]

print("init book keeping")
# dump config
config_dict = vars(args)
config_dict['nb_cubes'] = nb_cubes
config_dict['nb_rot'] = nb_task_space_rot
config_dict['start_cube_idx'] = start_cube_idx
config_dict['is_toy_env_2'] = args.toy_env_2
with open(save_dir + '/config.json', "w", encoding="utf8") as handle:
     json.dump(config_dict, handle)

nb_episodes = int(args.nb_episodes * 100000)
epoch_step = args.epoch_step * 100000
if args.toy_env_2 is True:
    env = ClassroomToyEnvV2(nb_dims=nb_dims, nb_cubes=nb_cubes, idx_first_cube=start_cube_idx)
else:
    env = ClassroomToyEnv(nb_dims=nb_dims, nb_cubes=nb_cubes, nb_task_space_rot=nb_task_space_rot)
env.reset()

epochs_score = []
epochs_time = []
epochs_episode_nb = []
epochs_comp_grid = []
comp_grid_at_teacher_updates = [env.get_cube_competence().astype(np.int8)]
train_tasks = []
train_rewards = []
episode_all_mastered = -1

# setup tasks generator
teacher_params['is_toy_env'] = True
if teacher_name == 'Oracle':
    Teacher = GaussianOracleTeacher(mins, maxs, teacher_params['window_step_vector'], seed=seed)
elif teacher_name == 'Random':
    Teacher = RandomTeacher(mins, maxs, seed=seed)
elif teacher_name == 'RIAC':
    Teacher = RIAC(mins, maxs, seed=seed, params=teacher_params)
elif teacher_name == 'ALP-GMM':
    Teacher = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
elif teacher_name == 'Covar-GMM':
    Teacher = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
elif teacher_name == 'EGT':
    Teacher = EGT(mins, maxs, seed=seed, params=teacher_params)
elif teacher_name == 'AGAIN':
    Teacher = AGAIN(mins, maxs, seed=seed, params=teacher_params)
elif teacher_name == 'ADR':
    Teacher = ADR(mins, maxs, seed=seed, params=teacher_params)
else:
    print('Unknown teacher')
    raise NotImplementedError



print('launching {} for {} on toy env with {} cubes and {} 90rot'.format(teacher_name, nb_episodes, nb_cubes, nb_task_space_rot))
# Main loop: collect experience in env and update/log each epoch
verbose = True
start_time = time.time()
for i in range(nb_episodes + 1):
    if (i % epoch_step) == 0:  # training epoch completed, record score
        epochs_time.append(time.time() - start_time)
        epochs_score.append(env.get_score())
        epochs_comp_grid.append(env.get_cube_competence().astype(np.int8))
        epochs_episode_nb.append(i)
        if nb_dims == 2:
            if verbose:
                print("it:{}, score:{}".format(i, epochs_score[-1]))
                print(epochs_comp_grid[-1])
        else:
            if verbose:
                print("it:{}, score:{}".format(i, epochs_score[-1]))

        if teacher_name == 'AGAIN' and i > 0:
            Teacher.send_test_info(epochs_comp_grid[-1].flatten())

    # sample task params
    task_params = copy.copy(Teacher.sample_task())
    assert type(task_params[0]) == np.float32
    train_tasks.append(task_params)

    if (i % 50) == 0:
        if env.get_score() == 100.0 and episode_all_mastered == -1:
            print('task space mastered at ep {} !!'.format(i))
            episode_all_mastered = i
    #print('ep:{},p={}'.format(i, task_params))

    reward = env.episode(task_params)
    is_teacher_updated = Teacher.update(np.array(task_params), reward)
    if is_teacher_updated:
        #print('teacher updated at {}'.format(i))
        comp_grid_at_teacher_updates.append(env.get_cube_competence().astype(np.int8))

    train_rewards.append(reward)

# Pickle data
with open(save_dir + '/env_params_save.pkl', 'wb') as handle:
    dump_dict = {'env_params_train': np.array(train_tasks).astype(np.float16),
                 'env_train_rewards': np.array(train_rewards).astype(np.float16),
                 'epochs_score': epochs_score,
                 'epochs_comp_grid': epochs_comp_grid,
                 'teacher_update_comp_grid': comp_grid_at_teacher_updates,
                 'epochs_episode_nb': epochs_episode_nb,
                 'time': epochs_time,
                 'ep_all_mastered': episode_all_mastered}
    dump_dict = Teacher.dump(dump_dict)
    if teacher_name == 'ALP-GMM' or (teacher_name == 'AGAIN' and args.use_alpgmm):
        dump_dict['tasks_alps'] = np.array(dump_dict['tasks_alps']).astype(np.float16)
        dump_dict['tasks_origin'] = np.array(dump_dict['tasks_origin']).astype(np.int8)

    if args.light is True:  # discard non essential saves
        light_dump_dict = {'epochs_score': epochs_score,
                           'epochs_comp_grid': epochs_comp_grid,
                           'epochs_episode_nb': epochs_episode_nb,
                           'time': epochs_time,
                           'ep_all_mastered': episode_all_mastered}
        dump_dict = light_dump_dict


    pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

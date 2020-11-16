import numpy as np
import pickle
import copy
from teachDRL.teachers.algos.riac import RIAC
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from teachDRL.teachers.algos.covar_gmm import CovarGMM
from teachDRL.teachers.algos.adr import ADR
from teachDRL.teachers.algos.random_teacher import RandomTeacher
from teachDRL.teachers.algos.oracle_teacher import OracleTeacher
from teachDRL.teachers.algos.oracle_teacher import GaussianOracleTeacher
from teachDRL.teachers.algos.egt import EGT
from teachDRL.teachers.algos.again import AGAIN
from teachDRL.teachers.utils.test_utils import get_test_set_name
from collections import OrderedDict

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
    #print('reconstructed param vector {}\n into {}'.format(param, param_dict)) #todo remove
    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        #print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)



class TeacherController(object):
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, seed=None, teacher_params={}, custom_test_param_vec=None):
        self.teacher = teacher

        self.nb_custom_tests = 0
        if custom_test_param_vec is not None:
            self.nb_custom_tests = len(custom_test_param_vec)
            custom_test_param_dicts = [param_vec_to_param_dict(param_env_bounds, vec) for vec in custom_test_param_vec]
            self.custom_test_env_list = custom_test_param_dicts
            print('number of generated custom tests {}'.format(len(self.custom_test_env_list)))

        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if len(bounds) == 2:
                mins.append(bounds[0])
                maxs.append(bounds[1])
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                mins.extend([bounds[0]] * bounds[2])
                maxs.extend([bounds[1]] * bounds[2])
            else:
                print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                exit(1)
        self.task_dim = len(mins)

        # setup tasks generator
        if teacher == 'Oracle':
            self.task_generator = GaussianOracleTeacher(mins, maxs, teacher_params['window_step_vector'], seed=seed)
        elif teacher == 'Random':
            self.task_generator = RandomTeacher(mins, maxs, seed=seed)
        elif teacher == 'RIAC':
            self.task_generator = RIAC(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'ALP-GMM':
            self.task_generator = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'EGT':
            self.task_generator = EGT(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'AGAIN':
            self.task_generator = AGAIN(mins, maxs, seed=seed, params=teacher_params)
        elif teacher == 'ADR':
            self.task_generator = ADR(mins, maxs, seed=seed, params=teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        self.test_mode = None
        if self.task_dim == 2:
            self.test_mode = "uniform"  # "fixed_set"
        else:
            self.test_mode = "fixed_set"

        test_param_vec = None
        if self.test_mode == "fixed_set":  # WARNING only works for hexagon env
            test_param_vec = np.array(pickle.load(open("teachDRL/teachers/test_sets/hexagon_test_set.pkl", "rb")))
            #name = get_test_set_name(self.param_env_bounds)
            print('fixed set of {} tasks loaded'.format(len(test_param_vec)))
            #self.test_env_list = pickle.load( open("teachDRL/teachers/test_sets/"+name+".pkl", "rb" ) )
            #print('fixed set of {} tasks loaded: {}'.format(len(self.test_env_list),name))
        elif self.test_mode == "uniform":
            # select <nb_test_episodes> parameters choosen uniformly in the task space
            nb_steps = int(nb_test_episodes ** (1/self.task_dim))
            print(maxs[0])
            d1 = np.linspace(mins[0], maxs[0], nb_steps, endpoint=True)
            d2 = np.linspace(mins[1], maxs[1], nb_steps, endpoint=True)
            test_param_vec = np.transpose([np.tile(d1, len(d2)), np.repeat(d2, len(d1))])  # cartesian product
        test_param_dicts = [param_vec_to_param_dict(param_env_bounds, vec) for vec in test_param_vec]
        self.test_env_list = test_param_dicts
        print('number of generated tests {}'.format(len(self.test_env_list)))
        # print(test_param_dicts)

        #data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

        self.custom_env_params_test = []
        self.custom_env_test_rewards = []
        self.custom_env_test_len = []

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if self.teacher != 'Oracle':
            reward = np.interp(reward, (-150, 350), (0, 1))
            self.env_train_norm_rewards.append(reward)
        self.task_generator.update(self.env_params_train[-1], reward)

    def record_test_episode(self, reward, ep_len):
        if self.test_ep_counter > self.nb_test_episodes: # it was a custom test
            self.custom_env_test_rewards.append(reward)
            self.custom_env_test_len.append(ep_len)
        else:
            self.env_test_rewards.append(reward)
            self.env_test_len.append(ep_len)

        if self.test_ep_counter == (self.nb_test_episodes + self.nb_custom_tests):
            self.test_ep_counter = 0
            if self.teacher == 'AGAIN':
                test_vec = np.interp(self.env_test_rewards[-self.nb_test_episodes:].copy(), (-150, 350), (0, 1))
                restart_learner = self.task_generator.send_test_info(test_vec)
                return restart_learner

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len,
                         'env_param_bounds': list(self.param_env_bounds.items())}
            if self.nb_custom_tests > 0:
                dump_dict['custom_env_params_test'] = self.custom_env_params_test
                dump_dict['custom_env_test_rewards'] = self.custom_env_test_rewards
                dump_dict['custom_env_test_len'] = self.custom_env_test_len
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def set_env_params(self, env):
        params = copy.copy(self.task_generator.sample_task())
        assert type(params[0]) == np.float32
        self.env_params_train.append(params)
        param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
        #print(param_dict)
        if 'gap_pos' in param_dict:
            env.set_environment(**param_dict)
        else:
            env.env.set_environment(**param_dict)
        return params

    def set_test_env_params(self, test_env, increment=True):
        if increment: self.test_ep_counter += 1
        if self.test_ep_counter > self.nb_test_episodes: # time for custom tests
            test_param_dict = self.custom_test_env_list[self.test_ep_counter - 1 - self.nb_test_episodes]
            test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
            if increment: self.custom_env_params_test.append(test_param_vec)
        else:
            test_param_dict = self.test_env_list[self.test_ep_counter-1]
            test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
            if increment: self.env_params_test.append(test_param_vec)

        # print('test param dict is: {}'.format(test_param_dict))

        #print('test param vector nb:{} is: {}'.format(self.test_ep_counter, test_param_vec))

        if 'gap_pos' in test_param_dict:
            test_env.set_environment(**test_param_dict)
        else:
            test_env.env.set_environment(**test_param_dict)
        return test_param_dict

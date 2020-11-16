from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box
from teachDRL.teachers.utils.dataset import BufferedDataset
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from collections import deque
from scipy.stats import multivariate_normal
import pickle
from sklearn.neighbors import NearestNeighbors
import os
import random


# FUNCTION PERFORMING THE KC-BASED STUDENT SELECTION PROCESS
def get_k_experts(current_student_param, current_test_vec, initial_test_vec_history, last_test_vec_history,
                  student_id_history, student_params, last_perfs, k=5, test_vec_idx=1, use_ground_truth=False, is_toy_env=False, is_v2=False):
    cur_student_param_idx = 0  # is env dependant, so we set it here
    # build student param vec history
    stud_param_names = None
    if is_toy_env and is_v2:
        stud_param_names = ['start_cube_idx']
    if is_toy_env and not is_v2:
        stud_param_names = ['nb_rot']
    if not is_toy_env and not is_v2:  # wc environment
        if 'nn' in student_params:
            stud_param_names = ['agent_type', 'leg_s', 'nn']
            cur_student_param_idx = 1
        else:  # legacy stuff TODO remove
            stud_param_names = ['agent_type']
    student_params_vec = []
    for i in range(len(student_id_history)):
        student_params_vec.append([student_params[p_name][i] for p_name in stud_param_names])

    if use_ground_truth:
        # build the student dataset with only identical morphologies
        student_param_history = []
        original_student_idx = []
        print(is_toy_env)
        # works for toy env and wc env
        for i, stud_param in enumerate(student_params_vec):
            if len(stud_param) == 1:
                if stud_param == current_student_param:
                    student_param_history.append(stud_param)
                    original_student_idx.append(i)
            elif len(stud_param) == 3:  # new wc enc setup
                if [stud_param[0], stud_param[2]] == [current_student_param[0], current_student_param[2]]:
                    print(stud_param[2])
                    student_param_history.append(stud_param[1])
                    original_student_idx.append(i)
            else:
                print('TODO IMPLEMENT')
                raise Exception
        print('added {} classroom student for gt knn'.format(len(student_param_history)))
        print(student_param_history)
        student_param_history = np.array(student_param_history).astype(np.float32)
        #print(student_param_history.shape)
        student_param_history = np.array(student_param_history).astype(np.float32).reshape(-1, 1)
        n_neighbors = min(k,len(student_param_history))
        if n_neighbors != k: print('reducing k to {} because not enough students in classroom'.format(n_neighbors))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(student_param_history)
        dist, idx = nbrs.kneighbors(np.array(current_student_param[cur_student_param_idx]).reshape(-1, 1))
        # map knn indices to original indexing
        og_idx = []
        # print(idx)
        for i in idx[0]:
            og_idx.append(original_student_idx[i])
        idx = [og_idx]
    else:
        #initial_test_vec_history = np.stack([np.array(vec) for vec in initial_test_vec_history], axis=0)
        initial_test_vec_history = np.array(initial_test_vec_history[test_vec_idx]).astype(np.float32)
        current_test_vec = np.array(current_test_vec)
        n_neighbors = min(k, len(student_id_history))
        if n_neighbors != k: print('reducing k to {} because not enough students in classroom'.format(n_neighbors))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(initial_test_vec_history)
        dist, idx = nbrs.kneighbors(np.array(current_test_vec).reshape(1,-1))
    dist = np.round(dist, 4)
    scores = []
    for i, idxx in enumerate(idx[0]):
        scores.append(np.round(sum(last_test_vec_history[idxx]),2))
    max_idx = np.argmax(scores)
    print('selecting curriculum priors for current student with params: {}'.format(current_student_param))
    for i, idxx in enumerate(idx[0]):
        #s_param_idx = original_student_idx[i]
        if i == max_idx:
            print('{}={}, dist: {}, score: {} stud_params:{}!!!!!!!!!!!!!!!!!!!!!'.format(student_id_history[idxx],
                                                                            last_perfs[idxx],
                                                                            dist[0][i], scores[i], [student_params[p_name][idxx] for p_name in stud_param_names]))
        else:
            print('{}={}, dist: {}, score: {} stud_params:{}'.format(student_id_history[idxx], last_perfs[idxx],
                                                      dist[0][i], scores[i], [student_params[p_name][idxx] for p_name in stud_param_names]))
    return student_id_history[idx[0][max_idx]]



# FUNCTION THAT EXTRACTS THE EXPERT CURRICULUM FROM THE TRAINING TRAJECTORY SELECTED BY THE PREVIOUS FUNCTION
def load_expert_trajectory(folder_path, alp_thr=0.2, max_steps=10000000, is_toy_env=False):
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
           or k == 'env_train_rewards' or 'tasks_origin':
            data[k] = v

    # 2 - pre-processing expert trajectory
    # removing low-alp gaussians
    processed_gmms_means = []
    processed_gmms_covs = []
    processed_gmms_mean_rew = []
    idx_removed_gmms = []
    step_nb = 0
    gmm_step = data['episodes'][0]
    for i, (gmm_means, gmm_covs, episode) in enumerate(zip(data["means"], data["covariances"], data['episodes'])):
        processed_gmm_means = []
        processed_gmm_covs = []
        #all_rewards = None
        if is_toy_env:
            all_rewards = data['env_train_rewards'][episode:episode + gmm_step]  # from gmm
        else:
            all_rewards = np.interp(data['env_train_rewards'][episode:episode + gmm_step], (-150, 350), (0, 1))  # from gmm
        all_gmm_idx = data['tasks_origin'][episode:episode + gmm_step]
        rewards = all_rewards[-50:]  # consider mean reward after some training on the GMM
        mean_reward = np.mean(rewards)
        high_lp_gmm_idx = []
        for j, (means, covs) in enumerate(zip(gmm_means, gmm_covs)):
            if means[-1] > alp_thr:  # last mean is ALP dimension
                high_lp_gmm_idx.append(j)
                # add gaussian
                processed_gmm_means.append(means)
                processed_gmm_covs.append(covs)
        if not processed_gmm_means == []:  # gmm not empty after pre-process, lets add it
            processed_gmms_means.append(processed_gmm_means)
            processed_gmms_covs.append(processed_gmm_covs)

            # reverse walk reward list, take only 50 rewards from high_alp gaussians only to compute mean
            high_lp_rewards = []
            for k, (r, gmm_idx) in enumerate(zip(all_rewards[::-1], all_gmm_idx[::-1])):
                if gmm_idx in high_lp_gmm_idx:
                    high_lp_rewards.append(r)
                if len(high_lp_rewards) == 50:
                    break
            #processed_gmms_mean_rew.append(mean_reward)
            processed_gmms_mean_rew.append(np.mean(high_lp_rewards))
        else:
            idx_removed_gmms.append(i)
        if not is_toy_env:
            step_nb += sum(data['env_train_len'][i * gmm_step:(i + 1) * gmm_step])
            if step_nb > max_steps:
                break
    print('idx of removed gmms ({}/{}) in expert traj: {}'.format(len(data['means']) - len(processed_gmms_means),
                                                                  len(data['means']),
                                                                  idx_removed_gmms))
    if not is_toy_env: print('number of steps: {}'.format(step_nb))
    return processed_gmms_means, processed_gmms_covs, processed_gmms_mean_rew

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]



# Classroom Expert Gaussian Teacher
# mins / maxs are vectors defining task space boundaries (ex: mins=[0,0,0] maxs=[1,1,1])
class AGAIN():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.classroom_filename = "student_history" if "classroom_filename" not in params else params['classroom_filename']
        self.classroom_portion = 100 if "classroom_portion" not in params else params['classroom_portion']
        self.use_alpgmm = False if "use_alpgmm" not in params else params['use_alpgmm']
        self.pre_test_epoch_idx = 2 if "pretrain_epochs" not in params else params['pretrain_epochs']
        self.restart_after_pretrain = False if "restart_after_pretrain" not in params else params['restart_after_pretrain']
        self.k = 5 if "k" not in params else params['k']
        self.random_expert = False if "random_expert" not in params else params['random_expert']
        self.nb_test_epochs = 0
        self.use_ground_truth = False if 'use_ground_truth' not in params else params['use_ground_truth']
        self.is_toy_env = False if "is_toy_env" not in params else params['is_toy_env']
        self.current_student_params = params['student_params']
        #self.decorelate_alpgmm = False if "decorelate_alpgmm" not in params else params['decorelate_alpgmm']

        self.nb_alpgmm_gaussians = None

        # setting up alpgmm for pre-test phase
        self.alpgmm = ALPGMM(mins, maxs, seed=seed, params=params)
        self.is_new_alpgmm = False  # boolean used to track alpgmm's periodic updates
        self.random_task_ratio = 0.1
        self.post_pre_test_task_ratio = 0.02 if "random_task_ratio" not in params else params["random_task_ratio"]
        self.in_end_rnd = self.post_pre_test_task_ratio if 'in_end_rnd' not in params else params['in_end_rnd']

        self.sampled_gaussian_idx = None

        self.stop_R = False if "stop_R" not in params else params['stop_R']
        self.nb_eps_after_R = 0

        self.expert_means,  self.expert_covs, self.expert_mean_rewards = None, None, None  # will be defined after pre test
        self.expert_type = "P" if "expert_type" not in params else params["expert_type"]
        self.r_list_len = 50 if "r_list_len" not in params else params["r_list_len"]
        self.tol_ratio = 1.0 if "tol_ratio" not in params else params["tol_ratio"]

        if self.expert_type == 'R':
            self.reward_list = deque(maxlen=self.r_list_len)

        self.expert_idx = -1
        self.episode_nb = 0
        self.current_means = None
        self.current_covs = None
        self.current_mean_r = None

        # Boring book-keeping
        #self._update()
        self.bk = {'cegt_k': self.k, 'cegt_pt': self.pre_test_epoch_idx, 'cegt_expert_type': self.expert_type,
                   'cegt_cf': self.classroom_filename, 'cegt_rap': self.restart_after_pretrain, 'stop_R':self.stop_R,
                   'cegt_covariances': [], 'cegt_means': [],
                   'cegt_episodes': [self.episode_nb], 'cegt_tasks_origin': [],
                   'cegt_nb_alpgmm_gaussians': [], 'cegt_expert_idx':[],
                   'cegt_test_vectors':[]}

        if self.pre_test_epoch_idx == 0:
            self.send_test_info(None, epoch_0=True)

    def send_test_info(self, test_vec, epoch_0=False):
        self.bk['cegt_test_vectors'].append(test_vec)
        #print('len test vec is')
        #print(len(test_vec))
        #print(test_vec.shape)
        if epoch_0:  #do not increment if called from init
            assert(self.random_expert or self.use_ground_truth)
        else:
            self.nb_test_epochs += 1
        if self.nb_test_epochs == self.pre_test_epoch_idx:  # time to find an expert from classroom
            self.bk['pre_test_vec'] = test_vec
            # load classroom history
            path = "teachDRL/data/elders_knowledge/{}.pkl".format(self.classroom_filename)
            print("loading from {}".format(path))
            is_v2 = False
            if "v2" in self.classroom_filename:
                is_v2 = True

            student_ids, initial_test_vectors_list, last_test_vector, last_perfs, student_params = pickle.load(open(path, "rb"))
            if self.classroom_portion != 100:
                # take a random sample subpart of classroom
                sample_len = int(len(student_ids) * (self.classroom_portion/100))
                print('using only {} classroom data sampled randomly'.format(sample_len))
                old_rnd_state = random.getstate()
                random.seed(self.seed)
                sampled_student_ids = random.sample(student_ids, sample_len)
                sampled_initial_test_vectors_list = []
                for kc_v in initial_test_vectors_list:
                    random.seed(self.seed)
                    sampled_initial_test_vectors_list.append(random.sample(kc_v, sample_len))
                random.seed(self.seed)
                sampled_last_test_vector = random.sample(last_test_vector, sample_len)
                random.seed(self.seed)
                sampled_last_perfs = random.sample(last_perfs, sample_len)
                if self.is_toy_env and is_v2:
                    random.seed(self.seed)
                    student_params['start_cube_idx'] = random.sample(student_params['start_cube_idx'], sample_len)
                else:
                    print('portion of non toy env v2 classroom is not yet supported')
                    exit(1)
                random.setstate(old_rnd_state)  # restore random state
                # set classroom to classroom sample
                initial_test_vectors_list = sampled_initial_test_vectors_list
                student_ids = sampled_student_ids
                last_test_vector = sampled_last_test_vector
                last_perfs = sampled_last_perfs

            if self.random_expert:
                print('choosing expert randomly !')
                expert_id = np.random.choice(student_ids)
            else:
                expert_id = get_k_experts(self.current_student_params, test_vec, initial_test_vectors_list, last_test_vector,
                                          student_ids, student_params, last_perfs, k=self.k,
                                          use_ground_truth=self.use_ground_truth, test_vec_idx=self.pre_test_epoch_idx-1,
                                          is_toy_env=self.is_toy_env, is_v2=is_v2)
            self.bk['selected_expert'] = expert_id
            print('expert selected is: {}'.format(expert_id))
            # loading expert
            folder_path = 'teachDRL/data/elders_knowledge/' + expert_id.rsplit('_s',1)[0] + '/' + expert_id
            print(folder_path)
            self.expert_means, self.expert_covs, self.expert_mean_rewards = load_expert_trajectory(folder_path, is_toy_env=self.is_toy_env)
            self._update()

            # add alpgmm gaussians
            if self.use_alpgmm and self.alpgmm.gmm is not None:
                self.current_means += self.alpgmm.gmm.means_.tolist()
                self.current_covs += self.alpgmm.gmm.covariances_.tolist()
                self.nb_alpgmm_gaussians = len(self.alpgmm.gmm.means_)
            return self.restart_after_pretrain
        return False

    def _update(self):
        if self.expert_type == 'P':  # Pool type, single GMM out of all expert GMMs
            #print('P-updating')
            self.current_means = [sub_item for sub_list in self.expert_means for sub_item in sub_list]  # flatten
            self.current_covs = [sub_item for sub_list in self.expert_covs for sub_item in sub_list]    # same
        elif self.expert_type == 'T':  # Time type, expert trajectory is stepped every 250 episodes
                #print('T-updating')
                self.expert_idx = min(self.episode_nb // 250, len(self.expert_means)-1)
                self.current_means = self.expert_means[self.expert_idx].copy()  # flatten
                self.current_covs = self.expert_covs[self.expert_idx].copy()
        elif self.expert_type == 'R':  # Reward type, expert traj is stepped when mean reward > to previous self
            #print('R-updating')
            self.expert_idx = min(self.expert_idx + 1, len(self.expert_means)-1)
            self.current_means = self.expert_means[self.expert_idx].copy()
            self.current_covs = self.expert_covs[self.expert_idx].copy()
            self.current_mean_r = self.expert_mean_rewards[self.expert_idx] * self.tol_ratio
            self.reward_list = deque(maxlen=self.r_list_len)
        else:
            print('Unknown expert type')
            exit(1)

    def update(self, task, reward):
        #print("current means: {}, covs {}".format(len(self.current_means), len(self.current_covs)))
        #print("expert_idx: {}".format(self.expert_idx))
        self.episode_nb += 1

        if self.nb_test_epochs < self.pre_test_epoch_idx:  # pre-test phase, only use alp-gmm
            self.is_new_alpgmm = self.alpgmm.update(task, reward)
            if self.is_new_alpgmm:
                self.bk['cegt_covariances'].append(self.alpgmm.gmm.covariances_.copy())
                self.bk['cegt_means'].append(self.alpgmm.gmm.means_.copy())
                self.bk['cegt_episodes'].append(self.episode_nb)
                self.bk['cegt_expert_idx'].append(self.expert_idx)
            return self.is_new_alpgmm

        just_updated_gmm = False
        # handle AGAIN-R/T to ALP-GMM transition after finishing expert curriculum
        if self.use_alpgmm and (self.expert_type == "R" or self.expert_type == 'T') and self.stop_R and self.expert_idx == (len(self.expert_means) - 1) and self.nb_alpgmm_gaussians is not None:
            if self.nb_eps_after_R == 0:  # when AGAIN reaches the end of the expert curriculum, it can change rnd sampling
                self.random_task_ratio = 0.1
                self.post_pre_test_task_ratio = self.in_end_rnd  # switch back to high-exploration strategy
                print('switching to rnd of {} since last IN idx reached'.format(self.in_end_rnd))
                if self.expert_type == 'R':
                    self.expert_type = "stoppedR"
                    self.bk['stoppedR_episode'] = self.episode_nb
                elif self.expert_type == 'T':
                    self.expert_type = "stoppedT"
                    self.bk['stoppedT_episode'] = self.episode_nb

        # handle AGAIN-R/T smooth re-update of last IN gaussian
        if self.use_alpgmm and (self.expert_type == "stoppedR" or self.expert_type == 'stoppedT') and self.stop_R and self.expert_idx == (len(self.expert_means) - 1) and self.nb_alpgmm_gaussians is not None:
            if self.nb_eps_after_R == 0:  # first time, init last IN GMM gaussian tracking to update ALP periodically
                self.last_IN_gaussians_alps = [deque(maxlen=100) for _ in range(len(self.current_means) - self.nb_alpgmm_gaussians)]
                self.added_since_fit = 0
                assert((len(self.current_means) - self.nb_alpgmm_gaussians) == len(self.expert_means[-1]))
                print('TIME TO START POST IN, last expert has len {} -->  {}'.format(len(self.expert_means[-1]),self.expert_means[-1]))
            elif self.added_since_fit == 100:  # time to re-update the final IN lps gmm
                print('last in update time')
                #print(self.last_IN_gaussians_alps)
                just_updated_gmm = True
                self.added_since_fit = 0
                for i, alp_window in enumerate(self.last_IN_gaussians_alps):
                    if len(alp_window) == 0:
                        self.current_means[i][-1] = 0.0
                    else:
                        self.current_means[i][-1] = np.mean(alp_window)
                # remove alp-gmm gaussians to fit update pipeline (they will be re-added
                self.current_means = self.current_means[:-self.nb_alpgmm_gaussians]
                self.current_covs = self.current_covs[:-self.nb_alpgmm_gaussians]
                print('post in update to {}'.format(self.current_means))

            if self.sampled_gaussian_idx < (len(self.current_means) - self.nb_alpgmm_gaussians):  # last task from IN
                #print('adding alp to IN idx {} out of {}'.format(self.sampled_gaussian_idx,len(self.current_means) - self.nb_alpgmm_gaussians))
                self.last_IN_gaussians_alps[self.sampled_gaussian_idx].append(self.alpgmm.alps[-1])
                self.added_since_fit += 1
            self.nb_eps_after_R += 1

        # handle IN-R to ALP-GMM transition after finishing expert curriculum
        if self.expert_type == "R" and self.stop_R and self.expert_idx == (len(self.expert_means) - 1):
            self.use_alpgmm = True
            self.nb_eps_after_R += 1
            if self.nb_eps_after_R == 250:  # after a long time in last expert index, change strategy
                self.expert_type = "stoppedR"
                self.random_task_ratio = 0.1
                self.bk['stoppedR_episode'] = self.episode_nb
                # replace last expert idx by alpgmm gaussians
                self.current_means = []
                self.current_covs = []
                self.nb_alpgmm_gaussians = len(self.alpgmm.gmm.means_)
                just_updated_gmm = True

        # PROCESS DATA FOR R or T variants
        if self.expert_type == 'R' and self.bk['cegt_tasks_origin'][-1] == 'egt':   # add reward to list if from egt
            self.reward_list.append(reward)
        # check whether a GMM update is necessary, depending on the expert type
        if (self.expert_type == 'T' and (self.episode_nb % 250) == 0)\
           or (self.expert_type == 'R' and len(self.reward_list) == self.r_list_len and np.mean(self.reward_list) >= self.current_mean_r):
            if self.expert_idx != (len(self.expert_means) - 1):  # if not already at the end of expert curricula
                self._update()
                just_updated_gmm = True

        if self.use_alpgmm:
            if just_updated_gmm and self.nb_alpgmm_gaussians is not None:  # expert changed, add alpgmm part
                self.current_means += self.alpgmm.gmm.means_.tolist()
                self.current_covs += self.alpgmm.gmm.covariances_.tolist()

            # send data to alpgmm
            self.is_new_alpgmm = self.alpgmm.update(task, reward)

            if self.is_new_alpgmm:
                # update current GMM by replacing old gaussians from alpgmm with new ones
                if self.nb_alpgmm_gaussians is not None:
                    # remove old gaussians
                    self.current_means = self.current_means[:-self.nb_alpgmm_gaussians]
                    self.current_covs = self.current_covs[:-self.nb_alpgmm_gaussians]
                # add new gaussians
                #print('adding stuff')
                self.current_means += self.alpgmm.gmm.means_.tolist()
                self.current_covs += self.alpgmm.gmm.covariances_.tolist()
                self.nb_alpgmm_gaussians = len(self.alpgmm.gmm.means_)
                just_updated_gmm = True

        # book-keeping
        if just_updated_gmm:
            self.bk['cegt_covariances'].append(self.current_covs.copy())
            self.bk['cegt_means'].append(self.current_means.copy())
            self.bk['cegt_episodes'].append(self.episode_nb)
            self.bk['cegt_expert_idx'].append(self.expert_idx)
            self.bk['cegt_nb_alpgmm_gaussians'].append(self.nb_alpgmm_gaussians)
        return just_updated_gmm

    def sample_task(self):
        new_task = None
        task_origin = None
        #print(self.episode_nb)
        # pre-test phase, only use alp-gmm
        if self.nb_test_epochs < self.pre_test_epoch_idx:
            #print('pre-test-task-sampling')
            if (self.episode_nb < 250) or (np.random.random() < self.random_task_ratio):
                # Random task sampling
                new_task = self.alpgmm.random_task_generator.sample()
                task_origin = 'random'
            else:
                # alp-gmm task sampling
                task_origin = 'alpgmm'
                alp_means = []
                for pos in self.alpgmm.gmm.means_:
                    alp_means.append(pos[-1])

                # 2 - Sample Gaussian proportionally to its mean ALP
                idx = proportional_choice(alp_means, eps=0.0)

                # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
                new_task = np.random.multivariate_normal(self.alpgmm.gmm.means_[idx], self.alpgmm.gmm.covariances_[idx])[:-1]
                new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
            self.bk['cegt_tasks_origin'].append(task_origin)
            return new_task

        #print(self.random_task_ratio)
        if self.use_alpgmm and np.random.random() < self.post_pre_test_task_ratio:
            # Random task sampling
            new_task = self.alpgmm.random_task_generator.sample()
            task_origin = 'random'
        else:
            # ALP-based task sampling

            # 1 - Retrieve the mean ALP value of each Gaussian in the GMM
            alp_means = []
            for means in self.current_means:
                alp_means.append(means[-1])

            # 2 - Sample Gaussian proportionally to its mean ALP
            idx = proportional_choice(alp_means, eps=0.0)
            self.sampled_gaussian_idx = idx

            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
            new_task = np.random.multivariate_normal(self.current_means[idx], self.current_covs[idx])[:-1]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
            task_origin = 'egt'
            if self.use_alpgmm and self.alpgmm.gmm is not None:
                if idx >= len(self.current_means) - self.nb_alpgmm_gaussians:
                    task_origin = 'alpgmm'
        #print(task_origin)
        # boring book-keeping
        self.bk['cegt_tasks_origin'].append(task_origin)

        return new_task

    def dump(self, dump_dict):
        self.bk['cegt_initial_expert_means'] = self.expert_means
        self.bk['cegt_initial_expert_covs'] = self.expert_covs
        self.bk['cegt_student_param'] = self.current_student_params
        if self.expert_type == 'R' or self.expert_type == "stoppedR":
            self.bk['cegt_initial_expert_mean_rewards'] = self.expert_mean_rewards
        dump_dict.update(self.bk)
        if self.use_alpgmm:
            dump_dict.update(self.alpgmm.bk)
        return dump_dict
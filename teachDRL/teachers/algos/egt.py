from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box
from teachDRL.teachers.utils.dataset import BufferedDataset
from teachDRL.teachers.algos.alp_gmm import ALPGMM
from collections import deque
from scipy.stats import multivariate_normal

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]



# Expert Gaussian Teacher
# mins / maxs are vectors defining task space boundaries (ex: mins=[0,0,0] maxs=[1,1,1])
class EGT():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        self.use_alpgmm = False if "use_alpgmm" not in params else params['use_alpgmm']
        #self.decorelate_alpgmm = False if "decorelate_alpgmm" not in params else params['decorelate_alpgmm']

        self.nb_alpgmm_gaussians = None
        if self.use_alpgmm:
            print("Using ALP-GMM with EGT")
            self.alpgmm = ALPGMM(mins, maxs, seed=seed, params=params)
            self.is_new_alpgmm = False  # boolean used to track alpgmm's periodic updates
            self.random_task_ratio = 0.02 if "random_task_ratio" not in params else params["random_task_ratio"]
            self.sampled_gaussian_idx = None

            self.stop_R = False if "stop_R" not in params else params['stop_R']
            self.nb_eps_after_R = 0


        assert('expert_gmms' in params)
        self.expert_means,  self.expert_covs, self.expert_mean_rewards = params['expert_gmms']
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
        self._update()
        self.bk = {'egt_covariances': [self.current_covs.copy()], 'egt_means': [self.current_means.copy()],
                   'egt_episodes': [self.episode_nb], 'egt_tasks_origin': [],
                   'egt_nb_alpgmm_gaussians': [self.nb_alpgmm_gaussians], 'egt_expert_idx':[self.expert_idx]}

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
        just_updated_gmm = False
        if self.use_alpgmm and self.expert_type == "R" and self.stop_R and self.expert_idx == (len(self.expert_means) - 1) and self.nb_alpgmm_gaussians is not None:
            self.nb_eps_after_R += 1
            if self.nb_eps_after_R == 250:  # after a long time in last expert index, change strategy
                self.expert_type = "stoppedR"
                self.random_task_ratio = 0.1
                self.current_means = []
                self.current_covs = []
                self.bk['stoppedR_episode'] = self.episode_nb
                just_updated_gmm = True

        # process new data
        if self.expert_type == 'R' and self.bk['egt_tasks_origin'][-1] == 'egt':   # add reward to list if from egt
            self.reward_list.append(reward)
        # check whether a GMM update is necessary, depending on the expert type
        if (self.expert_type == 'T' and (self.episode_nb % 250) == 0)\
           or (self.expert_type == 'R' and len(self.reward_list) == self.r_list_len and np.mean(self.reward_list) > self.current_mean_r):
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

        # # smoothly update the ALP value of expert gaussians if at last IEC index
        # if self.expert_idx == (len(self.expert_means) - 1) and self.use_alpgmm and self.nb_alpgmm_gaussians is not None:
        #     if self.expert_type == 'T' or (self.expert_type == 'R' and np.mean(self.reward_list) > self.current_mean_r):
        #         if self.bk['egt_tasks_origin'][-1] == 'egt':
        #             assert(self.sampled_gaussian_idx < (len(self.current_means) - self.nb_alpgmm_gaussians))
        #             cur_alp = self.current_means[self.sampled_gaussian_idx][-1] # update alp of corresponding Gaussian
        #             self.current_means[self.sampled_gaussian_idx][-1] = cur_alp * (49/50) + (self.alpgmm.alps[-1]/50)

        # book-keeping
        if just_updated_gmm:
            self.bk['egt_covariances'].append(self.current_covs.copy())
            self.bk['egt_means'].append(self.current_means.copy())
            self.bk['egt_episodes'].append(self.episode_nb)
            self.bk['egt_expert_idx'].append(self.expert_idx)
            self.bk['egt_nb_alpgmm_gaussians'].append(self.nb_alpgmm_gaussians)

    def sample_task(self):
        new_task = None
        task_origin = None
        if self.use_alpgmm and np.random.random() < self.random_task_ratio:
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
        self.bk['egt_tasks_origin'].append(task_origin)

        return new_task

    def dump(self, dump_dict):
        self.bk['egt_initial_expert_means'] = self.expert_means
        self.bk['egt_initial_expert_covs'] = self.expert_covs
        if self.expert_type == 'R' or self.expert_type == "stoppedR":
            self.bk['egt_initial_expert_mean_rewards'] = self.expert_mean_rewards
        dump_dict.update(self.bk)
        if self.use_alpgmm:
            dump_dict.update(self.alpgmm.bk)
        return dump_dict
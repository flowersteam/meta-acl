import numpy as np
from gym.spaces import Box
from collections import deque

class OracleTeacher():
    def __init__(self, mins, maxs, window_step_vector,
                 seed=None, reward_thr=230, step_rate=50):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)
        self.window_step_vector = window_step_vector
        self.reward_thr = reward_thr
        self.step_rate = step_rate
        self.window_range = (self.maxs - self.mins) / 6
        self.window_pos = np.zeros(len(self.mins), dtype=np.float32)  # stores bottom left point of window
        for i, step in enumerate(self.window_step_vector):
            if step > 0: # if step is positive, we go from min to max (and thus start at min)
                self.window_pos[i] = self.mins[i]
            else: # if step is negative, we go from max to min (and thus start at max - window_range)
                self.window_pos[i] = self.maxs[i] - self.window_range[i]

        self.train_rewards = []
        print("window range:{} \n position:{}\n step:{}\n"
              .format(self.window_range, self.window_pos, self.window_step_vector))

    def update(self, task, reward):
        self.train_rewards.append(reward)
        if len(self.train_rewards) == self.step_rate:
            mean_reward = np.mean(self.train_rewards)
            self.train_rewards = []
            if mean_reward > self.reward_thr:
                for i,step in enumerate(self.window_step_vector):
                    if step > 0:  # check if not stepping over max
                        self.window_pos[i] = min(self.window_pos[i] + step, self.maxs[i] - self.window_range[i])
                    elif step <= 0: # check if not stepping below min
                        self.window_pos[i] = max(self.window_pos[i] + step, self.mins[i])
                print('mut stump: mean_ret:{} window_pos:({})'.format(mean_reward, self.window_pos))

    def sample_task(self):
        task = np.random.uniform(self.window_pos, self.window_pos+self.window_range).astype(np.float32)
        #print(task)
        return task

    def dump(self, dump_dict):
        return dump_dict


class GaussianOracleTeacher():
    def __init__(self, mins, maxs, step_vector,
                 seed=None, reward_thr=230, std=0.05, step_rate=50):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)

        assert(len(mins) == 2)
        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)
        self.step_vector = step_vector
        self.reward_thr = reward_thr
        self.std = std
        self.step_rate = 50

        # build_expert_traj
        # means_h = np.arange(self.mins[0], self.maxs[0] + self.step_vector[0], self.step_vector[0])
        # means_s = np.arange(self.maxs[1], self.mins[1] + self.step_vector[1], self.step_vector[1])
        means_h = np.linspace(self.mins[0], self.maxs[0], self.step_rate, endpoint=True)
        means_s = np.linspace(self.maxs[1], self.mins[1], self.step_rate, endpoint=True)
        self.means = np.vstack((means_h, means_s)).T
        print(len(self.means))
        self.mean_idx = 0
        self.covar = np.array([[std, 0],
                               [0, std]])

        self.train_rewards = deque(maxlen=self.step_rate)
        print("last pos:{} \n first pos:{}\n step:{}\n"
              .format(self.means[-1], self.means[0], self.step_vector))
        self.task_cpt = 0
        self.update_episodes = []
    def update(self, task, reward):
        self.train_rewards.append(reward)
        if len(self.train_rewards) == self.step_rate:
            mean_reward = np.mean(self.train_rewards)
            if mean_reward >= 230.0:
                self.update_episodes.append(self.task_cpt)
                self.mean_idx = min(self.mean_idx + 1, len(self.means) - 1)
                self.train_rewards = deque(maxlen=self.step_rate)
                print('ep {}:mut stump: mean_ret:{} pos:({})'.format(self.task_cpt, mean_reward, self.means[self.mean_idx]))

    def sample_task(self):
        task = np.random.multivariate_normal(self.means[self.mean_idx],[[self.std,0],[0,self.std]]).astype(np.float32)
        task = np.clip(task, self.mins, self.maxs).astype(np.float32)
        self.task_cpt += 1
        return task

    def dump(self, dump_dict):
        dump_dict['oracle_episodes'] = self.update_episodes
        dump_dict['oracle_means'] = self.means
        return dump_dict
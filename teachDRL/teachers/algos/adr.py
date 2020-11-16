import numpy as np
from gym.spaces import Box
from collections import deque

# Automatic Domain Randomization, see https://arxiv.org/abs/1910.07113 for details
class ADR():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.nb_dims = len(self.mins)

        # initial calibrating task
        self.initial_task = np.array([0]*len(mins) if "initial_task" not in params else params["initial_task"])
        print('initial task is {}'.format(self.initial_task))

        # Boundary sampling probability p_r
        self.bound_sampling_p = 0.5 if "boundary_sampling_p" not in params else params["boundary_sampling_p"]

        # ADR step size
        self.step_size = [0.02]*self.nb_dims if "step_size" not in params else params["step_size"]

        # Increase threshold
        self.reward_threshold = 230 if "reward_thr" not in params else params["reward_thr"]
        self.is_toy_env = False if "is_toy_env" not in params else params['is_toy_env']
        print('is toy env: {}'.format(self.is_toy_env))
        if not self.is_toy_env:
            self.reward_threshold = np.interp(self.reward_threshold, (-150, 350), (0, 1))
        # max queue length
        self.window_len = 10 if "queue_len" not in params else params['queue_len']
        hyperparams = locals()
        # Set initial task space to predefined calibrated task
        self.cur_mins = np.array(self.initial_task, dtype=np.float32)  # current min bounds
        self.cur_maxs = np.array(self.initial_task, dtype=np.float32)  # current max bounds
        self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)

        # Init queues, one per task space dimension
        self.min_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]
        self.max_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]

        # Boring book-keeping
        self.episode_nb = 0
        print(hyperparams)
        self.bk = {'task_space': [(self.cur_mins.copy(),self.cur_maxs.copy())],
                   'episodes': [],
                   'ADR_hyperparams':hyperparams}

    def update(self, task, reward):
        self.episode_nb += 1
        # check for updates
        for i, (min_q, max_q, cur_min, cur_max) in enumerate(zip(self.min_queues, self.max_queues, self.cur_mins, self.cur_maxs)):
            if task[i] == cur_min:  # if the proposed task has the i^th dimension set to min boundary
                min_q.append(reward)
                if len(min_q) == self.window_len and np.mean(min_q) >= self.reward_threshold:  # decrease boundary
                    self.cur_mins[i] = max(self.cur_mins[i] - self.step_size[i], self.mins[i])
                    self.min_queues[i] = deque(maxlen=self.window_len)  # reset queue
            if task[i] == cur_max:  # if the proposed task has the i^th dimension set to max boundary
                max_q.append(reward)
                if len(max_q) == self.window_len and np.mean(max_q) >= self.reward_threshold:  # decrease boundary
                    self.cur_maxs[i] = min(self.cur_maxs[i] + self.step_size[i], self.maxs[i])
                    self.max_queues[i] = deque(maxlen=self.window_len)  # reset queue

        prev_cur_mins, prev_cur_maxs = self.bk['task_space'][-1]
        #print(self.bk['task_space'][-1])
        if (prev_cur_mins != self.cur_mins).any() or (prev_cur_maxs != self.cur_maxs).any():  # were boundaries changed ?
            self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)
            # book-keeping only if boundaries were updates
            self.bk['task_space'].append((self.cur_mins.copy(), self.cur_maxs.copy()))
            self.bk['episodes'].append(self.episode_nb)
            print(self.bk['task_space'][-1])

    def sample_task(self):
        new_task = self.task_space.sample()
        if np.random.random() < self.bound_sampling_p:  # set random dimension to min or max bound
            idx = np.random.randint(0, self.nb_dims)
            is_min_max_capped = np.array([self.cur_mins[idx] == self.mins[idx], self.cur_maxs[idx] == self.maxs[idx]])
            if not is_min_max_capped.all():  # both min and max bounds can increase, choose extremum randomly
                if np.random.random() < 0.5:  # skip min bound if already
                    new_task[idx] = self.cur_mins[idx]
                else:
                    new_task[idx] = self.cur_maxs[idx]
            elif not is_min_max_capped[0]:
                new_task[idx] = self.cur_mins[idx]
            elif not is_min_max_capped[1]:
                new_task[idx] = self.cur_maxs[idx]
        return new_task

    # def sample_task(self):
    #     new_task = self.task_space.sample()
    #     if np.random.random() < self.bound_sampling_p:  # set random dimension to min or max bound
    #         idx = np.random.randint(0,self.nb_dims)
    #         if self.cur_mins[idx] != self.mins[idx]:
    #             # if both min and max bounds can still increase, set random dimension to either min or max
    #             if np.random.random() < 0.5:  # skip min bound if already
    #                 new_task[idx] = self.cur_mins[idx]
    #             else:
    #                 new_task[idx] = self.cur_maxs[idx]
    #     return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict

if __name__ == '__main__':
    adr = ADR([0,0],[3,6],params={'initial_task':[0,6], 'step_size':[0.05, 0.1]})
    for i in range(5000):
        t = adr.sample_task()
        adr.update(t,250)
        print(i)
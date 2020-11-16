import numpy as np
from teachDRL.teachers.algos.alp_gmm import BufferedDataset
import time
import math
import random

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]


# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPComputer():
    def __init__(self, task_size, max_size=None, buffer_size=500, return_alp=True):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)
        self.return_alp = return_alp
        print(self.return_alp)

    def compute_alp(self, task, reward):
        alp = 0
        lp = 0
        if len(self.alp_knn) > 5:
            # Compute absolute learning progress for new task

            # 1 - Retrieve closest previous task
            dist, idx = self.alp_knn.nn_y(task)

            # 2 - Retrieve corresponding reward
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])

            # 3 - Compute alp as absolute difference in reward
            lp = reward - closest_previous_task_reward
            alp = np.abs(lp)

        # Add to database
        self.alp_knn.add_xy(reward, task)
        if self.return_alp:
            return alp
        else:  # return positive lp
            return max(0, lp)

class LPReplayBuffer:
    """
    experience replay buffer with Episode-wide Learning Progress based sampling
    """
    def __init__(self, obs_dim, act_dim, size, task_dim, ep_max_len=2000, alpha=1.0, use_alp=True, reward_based=False):
        print("Replay Sampling Activated with alpha= {}".format(alpha))


        self.obs1_buf = np.zeros([ep_max_len, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([ep_max_len, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([ep_max_len, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(ep_max_len, dtype=np.float32)
        self.done_buf = np.zeros(ep_max_len, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.ep_max_len = ep_max_len
        self.reward_based = reward_based

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Init ALP computer
        self.alp_computer = EmpiricalALPComputer(task_dim, return_alp=use_alp)

        # Init binary heap
        self.heap_buffer = Experience(self.max_size, alpha)

        self.current_task = None

    def start_task(self, task):
        assert(self.current_task is None)
        self.current_task = task

    def end_task(self, reward, ep_len):
        reward = np.interp(reward, (-150, 350), (0, 1))
        if not self.reward_based:
            lp = self.alp_computer.compute_alp(self.current_task, reward)
        #print("learning progress of episode: {}".format(lp))
        # add transitions to heap buffer
        for i in range(self.ptr):
            if self.reward_based:
                self.heap_buffer.add([self.obs1_buf[i], self.obs2_buf[i], self.acts_buf[i],
                                      self.rews_buf[i], self.done_buf[i]], reward)
            else:
                self.heap_buffer.add([self.obs1_buf[i], self.obs2_buf[i], self.acts_buf[i],
                                      self.rews_buf[i], self.done_buf[i]], lp)

        self.obs1_buf = np.zeros([self.ep_max_len, self.obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([self.ep_max_len, self.obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([self.ep_max_len, self.act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(self.ep_max_len, dtype=np.float32)
        self.done_buf = np.zeros(self.ep_max_len, dtype=np.float32)
        self.ptr = 0
        self.current_task = None


    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = self.ptr+1
        assert(self.ptr <= self.ep_max_len)
        #self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        data, idxs = self.heap_buffer.select(batch_size)
        for i,d in enumerate(data):
            if d is None:
                print('NONE NONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(d)
                print(idxs[i])
                print('NONE NONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                data.pop(i)  # removing None as a temporary fix


        obs1 = np.array([transition_info[0] for transition_info in data], dtype=np.float32)
        obs2 = np.array([transition_info[1] for transition_info in data], dtype=np.float32)
        acts = np.array([transition_info[2] for transition_info in data], dtype=np.float32)
        rews = np.array([transition_info[3] for transition_info in data], dtype=np.float32)
        done = np.array([transition_info[4] for transition_info in data], dtype=np.float32)
        #print('buf_size:{} prop sampling time: {}, w/o array {}'.format(self.heap_buffer.tree.filled_size(),end-start, end0-start))
        #print(idxs[:10])
        #print('SHHHHHHHHHHHHHHHHHHHAPU')
        #print(obs1.shape)
        #print(type(obs1))
        #print(type(obs1[0]))
        #print(type(obs1[0][0]))
        #print(obs2[0])
        return dict(obs1=obs1,
                    obs2=obs2,
                    acts=acts,
                    rews=rews,
                    done=done)

class Experience(object):
    """ The class represents prioritized experience replay buffer.
    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.
    see https://arxiv.org/pdf/1511.05952.pdf .
    """

    def __init__(self, memory_size, alpha):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha
        self.bonus_priority = 999  # add bonus priority for transitions that were never sampled
        self.epsilon_priority = 0.000001
        if self.alpha == 0: # revert to full uniform
            self.bonus_priority = 0

    def add(self, data, priority):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        if priority/100 >= self.bonus_priority and self.alpha != 0.0:
            print('WARNING, YOUR BONUS PRIORITY IS TOO LOW')
            exit(0)
        self.tree.add(data, priority ** self.alpha + self.bonus_priority + self.epsilon_priority)

    def select(self, batch_size):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        #if self.tree.filled_size() < batch_size:
        #    print('CALLING REPLAY SAMPLING WHEN NOT FULL ENOUGH')
        #    #return None, None

        out = []
        indices = []
        #weights = []
        priorities = []
        avoid_resampling = False
        for _ in range(batch_size):
            r = random.random()
            #return (idx, self.tree[idx], self.data[dataIdx])
            data, priority, index = self.tree.find(r)
            #index, priority, data = self.tree.find(r)
            #print(index)
            #print("d: {}, \n priority: {}, \n index: {}".format(data, priority, index))
            priorities.append(priority)
            #weights.append((1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            if avoid_resampling: self.priority_update([index], [self.epsilon_priority])  # To avoid resampling same transition too much

        for i in range(len(priorities)):
            if priorities[i] >= self.bonus_priority: # remove priority bonus
                priorities[i] -= self.bonus_priority
                self.priority_update([indices[i]],[priorities[i]])

        # avoid resampling part self.priority_update(indices, priorities)  # Revert priorities
        #weights /= max(weights)  # Normalize for stability
        return out, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class oldnewSumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.val_update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def filled_size(self):
        return self.n_entries

    # update priority
    def val_update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def find(self, s, norm=True):
        if norm:
            s *= self.tree[0]
        print('tree0: {} norm_s: {}'.format(self.tree[0], s))
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def print_tree(self):
        pass
        # tree_level = math.ceil(math.log((self.capacity-1) + 1, 2)) + 1
        # for k in range(1, tree_level + 1):
        #     for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
        #         print(self.tree[j], end=' ')
        #     print()


class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:
            return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
                        2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left:
            return self._find(value, 2 * index + 1)
        else:
            return self._find(value - left, 2 * (index + 1))

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size


if __name__ == '__main__':
    # replay_buffer = Experience(20,1.0)
    # # add transitions to buffer
    # for i in range(2000):
    #     replay_buffer.add(42+i, 0.0)
    #     # if i ==19:
    #     #     replay_buffer.add([0,1,2,3,4], 0.5)
    #     #     replay_buffer.add([0, 1, 2, 3, 4], 0.2)
    #
    # replay_buffer.tree.print_tree()
    # # sampling batches from tree:
    # for i in range(2000):
    #     data, inds = replay_buffer.select(2)
    #     print(inds)
    #     # print(type(data))
    #     # print(type(data[0]))
    # replay_buffer.tree.print_tree()
    #
    #

    # speed test
    replay_buffer = Experience(2000000, 0.0)
    # add transitions to buffer
    for i in range(20000):
        for i in range(100000):
            replay_buffer.add([np.random.random()], np.random.random())
            # if i ==19:
            #     replay_buffer.add([0,1,2,3,4], 0.5)
            #     replay_buffer.add([0, 1, 2, 3, 4], 0.2)

        # sampling batches from tree:
        buffer_sampling_time = 0
        for i in range(200):
            start = time.time()
            data, inds = replay_buffer.select(1000)
            end = time.time()
            buffer_sampling_time += end-start
        print('buf_size:{} prop sampling time: {}'.format(replay_buffer.tree.filled_size(), buffer_sampling_time/200))
            # print(type(data))
            # print(type(data[0]))






"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import pickle


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

    def save(self, saveto):
        with open(saveto+"scaler.pkl", 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


class Buffer:

    def __init__(self, buffer_size, obs_dim, act_dim):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = None
        self.act = None
        self.rewards = None
        self.obs_next = None

    def append(self, trajectories):
        """
        Next observation at the end of each trajectory is taken as zeros proportional to the observation dimension,
        which is biased because not every observation at the last time step is a terminal state.
        Maybe we should scrape the last quadruples at the end of each trajectory.
        :param trajectories:
        :return:
        """
        if self.obs is None:
            self.obs = np.concatenate([t['observes'] for t in trajectories])   # last tuple doesn't correspond to next observation.
            self.act = np.concatenate([t['actions'] for t in trajectories])
            self.rewards = np.concatenate([t['rewards'] for t in trajectories])
            self.obs_next = np.concatenate([np.append(t['observes'][1:], np.zeros((1, self.obs_dim)), axis=0) for t in trajectories])
        else:
            np.append(np.concatenate([t['observes'] for t in trajectories]),self.obs, axis=0) # append (E*T, obs_dim)
            np.append(np.concatenate([t['actions'] for t in trajectories]), self.act, axis=0)
            np.append(np.concatenate([t['rewards'] for t in trajectories]), self.rewards, axis=0)
            np.append(np.concatenate([np.append(t['observes'][1:], np.zeros((1, self.obs_dim)), axis=0) for t in trajectories]), self.obs_next, axis=0)

        if self.obs.shape[0] > self.buffer_size:
            cutoff = int(self.buffer_size*np.random.random())
            self.obs = self.obs[:cutoff]
            self.act = self.act[:cutoff]
            self.rewards = self.rewards[:cutoff]
            self.obs_next = self.obs_next[:cutoff]

    def sample(self, num_samples):
        """
        For Q critic training. Normally would take E*T number of samples
        :param num_samples: actual number of samples is the minimum of num_samples and total_samples that reside in this buffer.
        :return: permuted, loose-tied, quadruples.
        """
        total_samples = self.obs.shape[0]
        num_samples  = min(num_samples, total_samples)
        permute = np.random.permutation(np.linspace(0, total_samples - 1, total_samples, dtype=int))
        permute = permute[:num_samples]
        return self.obs[permute], self.act[permute], self.rewards[permute], self.obs_next[permute]


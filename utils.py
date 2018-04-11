"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import pandas as pd


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
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means

    def save(self, saveto):
        with open(saveto + "scaler.pkl", 'wb') as output:
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
        :param trajectories: Pay attention we only use scaled rewards
        :return:
        """
        if self.obs is None:
            self.obs = np.concatenate(
                [t['observes'] for t in trajectories])  # last tuple doesn't correspond to next observation.
            self.act = np.concatenate([t['actions'] for t in trajectories])
            self.rewards = np.concatenate([t['scaled_rewards'] for t in trajectories])
            self.obs_next = np.concatenate(
                [np.append(t['observes'][1:], np.zeros((1, self.obs_dim)), axis=0) for t in trajectories])
        else:
            self.obs = np.append(np.concatenate([t['observes'] for t in trajectories]), self.obs,
                                 axis=0)  # append (E*T, obs_dim)
            self.act = np.append(np.concatenate([t['actions'] for t in trajectories]), self.act, axis=0)
            self.rewards = np.append(np.concatenate([t['scaled_rewards'] for t in trajectories]), self.rewards, axis=0)
            self.obs_next = np.append(np.concatenate(
                [np.append(t['observes'][1:], np.zeros((1, self.obs_dim)), axis=0) for t in trajectories]),
                self.obs_next, axis=0)

        if self.obs.shape[0] > self.buffer_size:
            cutoff = int(self.buffer_size * np.random.random())
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
        num_samples = min(num_samples, total_samples)
        permute = np.random.permutation(np.linspace(0, total_samples - 1, total_samples, dtype=int))
        permute = permute[:num_samples]
        return self.obs[permute], self.act[permute], self.rewards[permute], self.obs_next[permute]

    def size(self):
        return self.rewards.shape[0]


class Plotter:

    def __init__(self, csv_logs, keys, legends):
        self.dfs = []
        if type(csv_logs) is list:
            for log in csv_logs:
                assert (os.path.exists(log))
                df = pd.read_csv(log)
                for key in keys:
                    assert (key in df.keys())
                self.dfs.append(df)
        else:
            self.dfs = [pd.read_csv(csv_logs)]

        self.keys = keys
        assert (len(csv_logs) == len(legends))
        self.legends = legends

    def plot(self, limit_episodes=None, saveto='./graph/plot.png'):
        f, axes = plt.subplots(1, len(self.keys), figsize=(16,9))
        for i, key in enumerate(self.keys):
            axes[i].set_xlabel('episodes')
            axes[i].set_ylabel(key)
            scale_x = 20
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
            axes[i].xaxis.set_major_formatter(ticks_x)
            plots = []
            labels = []
            for j, df in enumerate(self.dfs):
                if limit_episodes is None:
                    p, = axes[i].plot(list(df[key]))
                else:
                    p, = axes[i].plot(list(df[key])[:limit_episodes])
                plots.append(p)
                labels.append(self.legends[j] + " " + key)
            axes[i].legend(plots, labels)
        plt.savefig(saveto)


if __name__ == "__main__":
    #Hopper-v2 comparison between Q-PROP and PPO
    plotter = Plotter(['./results/QPROP/Hopper-v2_Default/2018-04-10_21_42_30/log.csv',
                       './results/offline-PPO/Hopper-v2_Default/2018-04-08_15_36_56/log.csv'], ['steps', 'rewards'],
                      ['Q-PROP', 'PPO'])
    plotter.plot(limit_episodes=15, saveto='./graph/hp_plot.png')

    # FetchReach-v0 comparison
    plotter = Plotter(['./results/QPROP/FetchReach-v0_Default/2018-04-11_17_05_54/log.csv',
                       './results/offline-PPO/FetchReach-v0_Default/2018-04-11_16_58_46/log.csv'], ['entropy', 'rewards'],
                      ['Q-PROP', 'PPO'])
    plotter.plot(saveto='./graph/fr_plot.png')

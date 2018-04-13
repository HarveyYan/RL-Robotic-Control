"""
Proximal Policy Optimization without eligibility trace;
batch update method.

Special thanks to Patrick Coady (pat-coady.github.io).
His implementation on PPO really helped me a lot.

"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import datetime
import argparse
import signal
import csv
import inspect
import shutil
import pickle
from matplotlib import ticker

from utils import Scaler

from Policy.PPOPolicy import ProximalPolicy
from Policy.NoTracePolicy import NoTracePolicy

from ValueFunc.BaselineValueFunc import ValueFunc
from ValueFunc.l2ValueFunc import l2TargetValueFunc
import scipy.signal

date_id = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_') + '/'


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class Experiment:

    def __init__(self, env_name, discount, num_iterations, lamb, animate, kl_target, show):
        self.env_name = env_name
        self.env = gym.make(env_name)
        if env_name == "FetchReach-v0":
            self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal', 'achieved_goal'])
        gym.spaces.seed(1234)
        self.obs_dim = self.env.observation_space.shape[0] + 1 # adding time step as feature
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.animate = animate
        self.episodes = 20
        self.killer = GracefulKiller()
        # self.policy = ProximalPolicy(self.obs_dim, self.act_dim, self.env.action_space, kl_target, discount=discount,
        #                              lamb=lamb)
        self.policy = NoTracePolicy(self.obs_dim, self.act_dim, self.env.action_space, kl_target, epochs=20)
        # using MC return would be more helpful
        self.value_func = l2TargetValueFunc(self.obs_dim, epochs=10)
        # self.value_func = ValueFunc(self.obs_dim, discount=discount, lamb=1)

        if not show:
            # save copies of file
            shutil.copy(inspect.getfile(self.policy.__class__), OUTPATH)
            shutil.copy(inspect.getfile(self.value_func.__class__), OUTPATH)
            shutil.copy(inspect.getfile(self.__class__), OUTPATH)

            self.log_file = open(OUTPATH + 'log.csv', 'w')
            self.write_header = True

        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)

        # Use of a scaler is crucial
        self.scaler = Scaler(self.obs_dim)
        self.init_scaler()

    def init_scaler(self):
        print('fitting scaler')
        observation_samples = []
        for i in range(5):
            observation = []
            obs = self.env.reset()
            observation.append(obs)
            obs = obs.astype(np.float64).reshape((1, -1))
            done = False
            step = 0
            while not done:
                obs = np.append(obs, [[step]], axis=1)  # add time step feature
                action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
                if self.env_name == "FetchReach-v0":
                    obs_new, reward, done, _ = self.env.step(action.reshape(-1))
                else:
                    obs_new, reward, done, _ = self.env.step(action)
                observation.append(obs_new)
                obs = obs_new.astype(np.float64).reshape((1, -1))
                step += 1e-3
            observation_samples.append(observation)
        observation_samples = np.concatenate(observation_samples, axis=0)
        # print(observation_samples.shape)
        self.scaler.update(observation_samples)

    def normalize_obs(self, obs):
        scale, offset = self.scaler.get()
        obs_scaled = (obs-offset)*scale
        self.scaler.update(obs.astype(np.float64).reshape((1, -1)))
        return obs_scaled

    def run_one_episode(self):
        """
        collect data only
        :param save:
        :param train_policy:
        :param train_value_func:
        :param animate:
        :return:
        """
        obs = self.env.reset()
        observes, actions, rewards = [],[],[]
        done = False
        step = 0
        while not done:
            if self.animate:
                self.env.render()
            obs = obs.astype(np.float64).reshape((1, -1))
            obs = self.normalize_obs(obs)
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            observes.append(obs)
            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            actions.append(action)
            if self.env_name == "FetchReach-v0":
                obs_new, reward, done, _ = self.env.step(action.reshape(-1))
            else:
                obs_new, reward, done, _ = self.env.step(action)
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            obs = obs_new
            step += 0.003

        return np.concatenate(observes), np.concatenate(actions), np.array(rewards)

    def discounted_sum(self, l, factor):
        discounted = []
        sum = 0
        for i in reversed(l):
            discounted.append(factor*sum+i)
            sum = factor*sum+i
        return np.array(list(reversed(discounted)))

    def run_policy(self, episodes):
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards = self.run_one_episode()
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards}

            # scale rewards
            # This step is really indispensable for the convergence of FetchReach environment
            if self.discount < 0.999:
                rewards = rewards*(1-self.discount)

            trajectory['values'] = self.value_func.predict(observes)
            trajectory['mc_return'] = self.discounted_sum(rewards, self.discount)

            trajectory['td_residual'] = rewards + self.discount*np.append(trajectory['values'][1:],0) - trajectory['values']
            trajectory['gae'] = self.discounted_sum(trajectory['td_residual'], self.discount*self.lamb)

            trajectories.append(trajectory)

        return trajectories

    def run_expr(self):
        ep_steps = []
        ep_rewards = []
        ep_entropy = []
        i = 0
        while i < self.num_iterations:
            trajectories = self.run_policy(20)
            i += len(trajectories)
            observes = np.concatenate([t['observes'] for t in trajectories])
            actions = np.concatenate([t['actions'] for t in trajectories])
            mc_returns = np.concatenate([t['mc_return'] for t in trajectories])
            # advantages = np.concatenate([t['td_residual'] for t in trajectories])
            advantages = np.concatenate([t['gae'] for t in trajectories])

            # normalize advantage estimates
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            value_func_loss = self.value_func.update(observes, mc_returns)
            policy_loss, kl, entropy, beta = self.policy.update(observes, actions, advantages)

            avg_rewards = np.sum(np.concatenate([t['rewards'] for t in trajectories])) / self.episodes
            avg_timesteps = np.average([len(t['rewards']) for t in trajectories])
            log = {}

            # compute statistics such as mean and std
            log['steps'] = avg_timesteps
            log['rewards'] = avg_rewards
            log['policy_loss'] = policy_loss
            log['kl'] = kl
            log['entropy'] = entropy
            log['value_func_loss'] = value_func_loss
            log['beta'] = beta

            # display
            print('episode: ', i)
            print('average steps: {0}, average rewards: {1}'.format(log['steps'], log['rewards']))
            for key in ['policy_loss', 'kl', 'entropy', 'beta', 'value_func_loss']:
                print('{:s}: {:.2g}'.format(key, log[key]))
            print('\n')
            ep_steps.append(log['steps'])
            ep_rewards.append(log['rewards'])
            ep_entropy.append(log['entropy'])


            # write to log.csv
            if self.write_header:
                fieldnames = [x for x in log.keys()]
                self.writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
                self.writer.writeheader()
                self.write_header = False
            self.writer.writerow(log)
            # we want the csv file to preserve information even if the program terminates earlier than scheduled.
            self.log_file.flush()

            # save model weights if stopped manually
            if self.killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                self.killer.kill_now = False

            # if (i+1)%20 == 0:
            #     print('episode: ', i+1)
            #     print('average steps', np.average(steps))
            #     print('average rewards', np.average(rewards))

        self.policy.save(OUTPATH)
        self.value_func.save(OUTPATH)
        self.scaler.save(OUTPATH)

        plt.figure(figsize=(12,9))

        if self.env_name.startswith('Fetch'):
            ax1 = plt.subplot(121)
            plt.xlabel('episodes')
            plt.ylabel('policy entropy')
            plt.plot(ep_entropy)
            scale_x = self.episodes
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
            ax1.xaxis.set_major_formatter(ticks_x)
        else:
            ax1 = plt.subplot(121)
            plt.xlabel('episodes')
            plt.ylabel('steps')
            plt.plot(ep_steps)
            scale_x = self.episodes
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
            ax1.xaxis.set_major_formatter(ticks_x)

        ax2 = plt.subplot(122)
        plt.xlabel('episodes')
        plt.ylabel('episodic rewards')
        plt.plot(ep_rewards)
        scale_x = self.episodes
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
        ax2.xaxis.set_major_formatter(ticks_x)

        plt.savefig(OUTPATH + 'train.png')

    def load_model(self, load_from):
        from tensorflow.python.tools import inspect_checkpoint as chkp

        # # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(load_from+'policy/policy.pl', tensor_name='', all_tensors=True, all_tensor_names=True)
        self.policy.load(load_from + 'policy/policy.pl')
        self.value_func.load(load_from + 'value_func/value_func.pl')

    def demonstrate_agent(self, load_from):
        self.load_model(load_from)
        with open(load_from + "scaler.pkl", 'rb') as file:
            self.scaler = pickle.load(file)
        self.animate = True
        for i in range(10):
            observes, actons, rewards = self.run_one_episode()
            ep_rewards = np.sum(rewards)
            ep_steps = len(rewards)
            print("Total steps: {0}, total rewards: {1}\n".format(ep_steps, ep_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of episodes to run', default=1000)
    parser.add_argument('-d', '--discount', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-k', '--kl_target', type=float, help='KL target', default=0.003)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation', default=0.98)
    parser.add_argument('-a', '--animate', type=bool, help='Render animation', default=False)
    parser.add_argument('-s', '--show', type=bool, help='Demonstrate a trainied agent', default=False)
    parser.add_argument('--show_dir', type=str, help='The saved model parameters to a trained agent')
    parser.add_argument('-m', '--message', type=str, help='Message/identifier for experiments', default="Default")
    args = parser.parse_args()

    if not args.show:
        print('training an agent anew, in environment: {}'.format(args.env_name))
        global OUTPATH
        OUTPATH = './results/offline-PPO/' + args.env_name + '_' + args.message + '/'  + date_id
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        del args.message
        del args.show_dir
        expr = Experiment(**vars(args))
        expr.run_expr()
    else:
        args.animate = True
        print('loading an agent: {}'.format(args.env_name))
        del args.message
        if args.show_dir is None:
            print('Needs to specify --show_dir when --show is active')
            exit()
        show_dir = args.show_dir # e.g. "./results/Hopper-v2/offline-PPO/2018-04-06_12_58_36/"
        del show_dir
        expr = Experiment(**vars(args))
        expr.demonstrate_agent(show_dir)


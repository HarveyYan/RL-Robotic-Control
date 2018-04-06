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

    def __init__(self, env_name, discount, num_iterations, lamb, animate, kl_target, demonstrate):
        self.env = gym.make(env_name)
        self.env = gym.wrappers.FlattenDictWrapper(self.env, ['observation', 'desired_goal', 'achieved_goal'])
        gym.spaces.seed(1234)
        self.obs_dim = self.env.observation_space.shape[0]
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

        if not demonstrate:
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
            while not done:
                action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
                obs_new, reward, done, _ = self.env.step(action.reshape(-1))
                observation.append(obs_new)
                obs = obs_new.astype(np.float64).reshape((1, -1))
            observation_samples.append(observation)
        observation_samples = np.concatenate(observation_samples, axis=0)
        # print(observation_samples.shape)
        self.scaler.update(observation_samples)

    def normalize_obs(self, obs):
        scale, offset = self.scaler.get()
        obs_scaled = (obs-offset)*scale
        self.scaler.update(obs.astype(np.float64).reshape((1, -1)))
        # return self.scaler.transform(obs)
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
            observes.append(obs)
            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            actions.append(action)
            obs_new, reward, done, _ = self.env.step(action.reshape(-1))
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            obs = obs_new
            step += 0.001

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
        i = 0
        while i < self.num_iterations:
            trajectories = self.run_policy(20)
            i += len(trajectories)
            observes = np.concatenate([t['observes'] for t in trajectories])
            actions = np.concatenate([t['actions'] for t in trajectories])
            mc_returns = np.concatenate([t['mc_return'] for t in trajectories])
            advantages = np.concatenate([t['td_residual'] for t in trajectories])

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
                    self.policy.save(OUTPATH)
                    self.value_func.save(OUTPATH)
                    self.scaler.save(OUTPATH)
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
        plt.subplot(121)
        plt.xlabel('episodes')
        plt.xticks(np.arange(len(ep_steps)), np.arange(len(ep_steps))*self.episodes)
        plt.ylabel('steps')
        plt.plot(ep_steps)

        plt.subplot(122)
        plt.xlabel('episodes')
        plt.xticks(np.arange(len(ep_rewards)), np.arange(len(ep_rewards)) * self.episodes)
        plt.ylabel('episodic rewards')
        plt.plot(ep_rewards)

        plt.savefig(OUTPATH + 'train.png')

    def load_model(self, load_from):
        from tensorflow.python.tools import inspect_checkpoint as chkp

        # # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(load_from+'policy/policy.pl', tensor_name='', all_tensors=True, all_tensor_names=True)
        self.policy.load(load_from + 'policy/policy.pl')
        self.value_func.load(load_from + 'value_func/value_func.pl')

    def demonstrate_agent(self):
        load_from = "./results/Hopper-v2/offline-PPO/2018-04-06_12_58_36/"
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
    parser.add_argument('-e', '--env_name', type=str, help='OpenAI Gym environment name', default="HumanoidStandup-v2")
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of episodes to run', default=1000)
    parser.add_argument('-d', '--discount', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-k', '--kl_target', type=float, help='KL target', default=0.003)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation', default=0.98)
    parser.add_argument('-a', '--animate', type=bool, help='Render animation', default=False)
    parser.add_argument('-s', '--demonstrate', type=bool, help='Demonstrate a trainied agent', default=False)
    args = parser.parse_args()

    if not args.demonstrate:
        print('training an agent anew')
        global OUTPATH
        OUTPATH = './results/' + args.env_name + '/' + 'offline-PPO/' + date_id
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        expr = Experiment(**vars(args))
        expr.run_expr()
    else:
        print('loading an agent: Hooper-v2')
        args.animate = True
        args.env_name = "Hopper-v2"
        expr = Experiment(**vars(args))
        expr.demonstrate_agent()


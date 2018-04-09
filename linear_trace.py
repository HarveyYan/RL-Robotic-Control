"""
Proximal Policy Optimization with eligibility traces on linear function approximator.

Special thanks to Patrick Coady (pat-coady.github.io).
His implementation on PPO really helped me a lot.

"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import Scaler
import os
import datetime
import argparse
import signal
import csv
import inspect
import shutil

from Policy.LinearPolicy import LinearPolicy
from ValueFunc.LinearValueFunc import LinearValueFunc

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

    def __init__(self, env_name, discount, num_iterations, lamb, animate, kl_target):
        self.env = gym.make(env_name)
        gym.spaces.seed(1234)
        self.obs_dim = self.env.observation_space.shape[0] + 1 # the use of time steps is beneficial
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.animate = animate
        self.killer = GracefulKiller()
        self.policy = LinearPolicy(self.obs_dim, self.act_dim, self.env.action_space, kl_target, discount=discount)
        self.value_func = LinearValueFunc(self.obs_dim, discount=discount)

        # save copies of file
        shutil.copy(inspect.getfile(self.policy.__class__), OUTPATH)
        shutil.copy(inspect.getfile(self.value_func.__class__), OUTPATH)
        shutil.copy(inspect.getfile(self.__class__), OUTPATH)

        self.log_file = open(OUTPATH + 'log.csv', 'w')
        self.write_header = True
        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)
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

    def run_one_epsisode(self, train_policy=True, train_value_func=True, animate=False):
        obs = self.env.reset()
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = self.normalize_obs(obs)
        obs = np.append(obs, [[0]], axis=1)  # add time step feature
        log = {
            'rewards': [],
            'policy_loss': [],
            'value_func_loss': [],
            'entropy': [],
            'beta': [],
            'kl': [],
            'advantage':[]
        }

        done = False
        step = 0
        while not done:
            if animate:
                self.env.render()
            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            step += 1e-3
            # print(action)
            obs_new, reward, done, _ = self.env.step(action)
            obs_new = obs_new.astype(np.float64).reshape((1, -1))
            obs_new = self.normalize_obs(obs_new)
            obs_new = np.append(obs_new, [[step]], axis=1)  # add time step feature

            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            log['rewards'].append(reward)

            # scale reward
            if self.discount < 0.999:
                reward *= (1-self.discount)

            # TD residual
            advantage = reward + self.discount * self.value_func.predict(obs_new) - self.value_func.predict(obs)
            advantage = advantage.astype(np.float64).reshape((1,))

            if train_value_func:
                value_func_loss = self.value_func.update(obs, advantage)
            if train_policy:
                policy_loss, kl, entropy, beta = self.policy.update(obs, action, advantage)

            if train_value_func and train_policy:
                log['policy_loss'].append(policy_loss)
                log['kl'].append(kl)
                log['entropy'].append(entropy)
                log['beta'].append(beta)
                log['value_func_loss'].append(value_func_loss)
                log['advantage'].append(advantage)

            obs = obs_new

        return log

    def run_expr(self):
        ep_steps = []
        ep_rewards = []
        for i in range(self.num_iterations):
            # trace vectors are emptied at the beginning of each episode

            # get more accurate value_func estimator
            for _ in range(5):
                self.value_func.init_trace()
                self.run_one_epsisode(train_value_func=True, train_policy=False, animate=False)

            self.policy.init_trace()
            self.value_func.init_trace()

            # run (and train) one trajectory
            log = self.run_one_epsisode(animate=self.animate)

            # compute statistics such as mean and std
            log['steps'] = len(log['rewards'])
            log['rewards'] = np.sum(log['rewards'])
            for key in ['policy_loss', 'kl', 'entropy', 'beta', 'value_func_loss', 'advantage']:
                log[key + '_mean'] = np.mean(log[key])
                log[key + '_std'] = np.std(log[key])
                del log[key]

            # display
            print('episode: ', i)
            print('total steps: {0}, episodic rewards: {1}'.format(log['steps'], log['rewards']))
            for key in ['policy_loss', 'kl', 'entropy', 'beta', 'value_func_loss', 'advantage']:
                print('{:s}: {:.2g}({:.2g})'.format(key, log[key + '_mean'], log[key + '_std']))
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
                    break
                self.killer.kill_now = False

            # if (i+1)%20 == 0:
            #     print('episode: ', i+1)
            #     print('average steps', np.average(steps))
            #     print('average rewards', np.average(rewards))

        # save weights
        self.policy.save(OUTPATH)
        self.value_func.save(OUTPATH)
        self.scaler.save(OUTPATH)

        plt.subplot(121)
        plt.xlabel('episodes')
        plt.ylabel('steps')
        plt.plot(ep_steps)

        plt.subplot(122)
        plt.xlabel('episodes')
        plt.ylabel('episodic rewards')
        plt.plot(ep_rewards)

        plt.savefig(OUTPATH + 'train.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of episodes to run', default=1000)
    parser.add_argument('-d', '--discount', type=float, help='Discount factor', default=0.9)
    parser.add_argument('-k', '--kl_target', type=float, help='KL target', default=0.003)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation', default=0.4)
    parser.add_argument('-a', '--animate', type=bool, help='Render animation or not', default=False)
    parser.add_argument('-m', '--message', type=str, help='Message/identifier for experiments', default="Default")
    args = parser.parse_args()

    global OUTPATH
    OUTPATH = './results/linear-trace/' + args.env_name+'_'+args.message + '/' + date_id
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    del args.message
    expr = Experiment(**vars(args))
    expr.run_expr()

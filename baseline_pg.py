"""
Baseline policy gradient with traces

Special thanks to Patrick Coady (pat-coady.github.io).
His implementation on PPO really helped me a lot.

"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import sklearn.preprocessing
import os
import datetime
import argparse

from Policy.BaselinePolicy import Policy
from ValueFunc.BaselineValueFunc import ValueFunc

date_id = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_') + '/'

class Experiment:

    def __init__(self, env_name, discount, num_iterations, lamb, animate):
        self.env = gym.make(env_name)
        gym.spaces.seed(1234)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.animate = animate
        self.policy = Policy(self.obs_dim, self.act_dim, self.env.action_space, discount=discount, lamb=lamb)
        self.value_func = ValueFunc(self.obs_dim, discount=discount, lamb=lamb)

        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)
        self.init_scaler()

    def init_scaler(self):
        print('fitting scaler')
        self.scaler = sklearn.preprocessing.StandardScaler()
        observation_samples = []
        for i in range(5):
            observation = []
            obs = self.env.reset()
            observation.append(obs)
            obs = obs.astype(np.float64).reshape((1, -1))
            done = False
            while not done:
                action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
                obs_new, reward, done, _ = self.env.step(action)
                observation.append(obs_new)
                obs = obs_new.astype(np.float64).reshape((1, -1))
            observation_samples.append(observation)
        observation_samples = np.concatenate(observation_samples, axis=0)
        # print(observation_samples.shape)
        self.scaler.fit(observation_samples)

    def normalize_obs(self, obs):
        return self.scaler.transform(obs)

    def run_one_epsisode(self, save=True, train=True, animate=False):
        obs = self.env.reset()
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = self.normalize_obs(obs)
        rewards = []
        done = False
        step = 0
        while not done:
            if animate:
                self.env.render()

            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            # print(action)
            obs_new, reward, done, _ = self.env.step(action)
            obs_new = obs_new.astype(np.float64).reshape((1, -1))
            obs_new = self.normalize_obs(obs_new)

            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            if train:
                advantage = reward + self.discount * self.value_func.predict(obs_new) - self.value_func.predict(obs)
                advantage = advantage.astype(np.float64).reshape((1,))

                self.policy.update(obs, action, advantage)
                self.value_func.update(obs, advantage)

            obs = obs_new
            step += 0.001

        if save:
            self.policy.save(OUTPATH)
            self.value_func.save(OUTPATH)

        return rewards


    def run_expr(self):
        steps = []
        undiscounted = []
        for i in range(self.num_iterations):
            # trace vectors are emptied at the beginning of each episode
            self.policy.init_trace()
            self.value_func.init_trace()

            rewards = self.run_one_epsisode(save=i == (self.num_iterations - 1), train=True, animate=self.animate)
            total_steps = len(rewards)
            print('episode: ', i)
            print('total steps: {0}, episode_reward: {1}'.format(total_steps, np.sum(rewards)))

            steps.append(total_steps)
            undiscounted.append(np.sum(rewards))

            # if (i+1)%20 == 0:
            #     print('episode: ', i+1)
            #     print('average steps', np.average(steps))
            #     print('average rewards', np.average(rewards))

        np.savetxt(OUTPATH + 'steps', steps, delimiter=',')
        np.savetxt(OUTPATH + 'rewards', undiscounted, delimiter=',')

        plt.subplot(121)
        plt.xlabel('episode')
        plt.ylabel('steps')
        plt.plot(steps)

        plt.subplot(122)
        plt.xlabel('episode')
        plt.ylabel('undiscounted rewards')
        plt.plot(undiscounted)

        plt.savefig(OUTPATH+'train.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, help='OpenAI Gym environment name', default="HumanoidStandup-v2")
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of episodes to run', default=1000)
    parser.add_argument('-d', '--discount', type=float, help='Discount factor', default=1.0)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation', default=0.8)
    parser.add_argument('-a', '--animate', type=bool, help='Render animation or not', default=False)
    args = parser.parse_args()

    global OUTPATH
    OUTPATH = './results/'+args.env_name+'/'+'baseline/'+date_id
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    expr = Experiment(**vars(args))
    expr.run_expr()

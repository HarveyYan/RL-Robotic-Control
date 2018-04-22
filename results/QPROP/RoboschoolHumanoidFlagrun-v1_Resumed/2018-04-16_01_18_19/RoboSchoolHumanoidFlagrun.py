"""
An initiative to experiment Q-PROP in a Roboschool environment,
namely the RoboschoolHumanoidFlagrun-v1, which is more challenging than
ordinary Mujoco environment.

Agents are trained from scratch using Q-PROP algorithm.

"""
import gym, roboschool
from OpenGL import GLU
import numpy as np

import os
import datetime
import argparse
import signal
import csv
import inspect
import shutil
import pickle
from matplotlib import ticker
import matplotlib.pyplot as plt

from utils import Scaler, Buffer
from Policy.QPropPolicy import QPropPolicy
from ValueFunc.l2ValueFunc import l2TargetValueFunc
from Critic.DetCritic import DeterministicCritic

date_id = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_') + '/'
OUTPATH = None

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


class Experiment:

    def __init__(self, discount, num_iterations, lamb, animate, kl_target, **kwargs):
        self.env_name = 'RoboschoolHumanoidFlagrun-v1'
        self.env = gym.make(self.env_name)
        gym.spaces.seed(1234) # for reproducibility
        self.obs_dim = self.env.observation_space.shape[0] + 1 # adding time step as feature
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.animate = animate

        self.buffer = Buffer(1000000, self.obs_dim, self.act_dim) # 1000000 is the size they have used in paper
        self.episodes = 20 # larger episodes can reduce variance
        self.killer = GracefulKiller()

        self.policy = QPropPolicy(self.obs_dim, self.act_dim, self.env.action_space, kl_target, epochs=20)
        self.critic = DeterministicCritic(self.obs_dim, self.act_dim, self.discount, OUTPATH)
        self.value_func = l2TargetValueFunc(self.obs_dim, epochs=10)

        if 'show' in kwargs and not kwargs['show']:
            # save copies of file
            shutil.copy(inspect.getfile(self.policy.__class__), OUTPATH)
            shutil.copy(inspect.getfile(self.value_func.__class__), OUTPATH)
            shutil.copy(inspect.getfile(self.critic.__class__), OUTPATH)
            shutil.copy(inspect.getfile(self.__class__), OUTPATH)

            self.log_file = open(OUTPATH + 'log.csv', 'w')
            self.write_header = True

        print('Observation dimension:', self.obs_dim)
        print('Action dimension:', self.act_dim)

        # The use of a scaler is crucial
        self.scaler = Scaler(self.obs_dim)
        self.init_scaler()

    def init_scaler(self):
        """
        Collection observations from 5 episodes to initialize Scaler.
        :return: a properly initialized scaler
        """
        print('Fitting scaler')
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
                obs_new, reward, done, _ = self.env.step(action.reshape(-1))
                observation.append(obs_new)
                obs = obs_new.astype(np.float64).reshape((1, -1))
                step += 1e-3
            observation_samples.append(observation)
        observation_samples = np.concatenate(observation_samples, axis=0)
        self.scaler.update(observation_samples)

    def normalize_obs(self, obs):
        """
        Transform and update the scaler on the fly.
        :param obs: Raw observation
        :return: normalized observation
        """
        scale, offset = self.scaler.get()
        obs_scaled = (obs-offset)*scale
        self.scaler.update(obs.astype(np.float64).reshape((1, -1)))
        return obs_scaled

    def run_one_episode(self):
        """
        collect a trajectory of (obs, act, reward, obs_next)
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
            obs = np.append(obs, [[step]], axis=1)  # add time step feature at normalized observation
            observes.append(obs)

            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            actions.append(action)
            obs_new, reward, done, _ = self.env.step(action.reshape(-1))
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            obs = obs_new
            step += 0.003

        return np.concatenate(observes), np.concatenate(actions), np.array(rewards)

    def discounted_sum(self, l, factor):
        """
        Discounted sum of return or advantage estimates along a trajectory.
        :param l: a list containing the values of discounted summed interest.
        :param factor: discount factor in the disc_sum case or discount*lambda for GAE
        :return: discounted sum of l with regard to factor
        """
        discounted = []
        sum = 0
        for i in reversed(l):
            discounted.append(factor*sum+i)
            sum = factor*sum+i
        return np.array(list(reversed(discounted)))

    def run_policy(self, episodes):
        """
        Gather a batch of trajectory samples.
        :param episodes: size of batch.
        :return: a batch of samples
        """
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards = self.run_one_episode()
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'scaled_rewards': rewards*(1-self.discount)}
            trajectories.append(trajectory)

        return trajectories

    def run_expr(self):
        ep_steps = []
        ep_rewards = []
        ep_entropy = []
        i = 0
        while i < self.num_iterations:
            trajectories = self.run_policy(20)
            # add to experience replay buffer
            self.buffer.append(trajectories)
            print('buffer size:', self.buffer.size())

            i += len(trajectories)

            # for E=20, T=50, the total number of samples would be 1000
            # In future needs to account for not uniform time steps per episode.
            # e.g. in Hopper-v2 environment not every episode has same time steps
            # E = len(trajectories)
            # num_samples = np.sum([len(t['rewards']) for t in trajectories])
            gradient_steps = np.sum([len(t['rewards']) for t in trajectories])

            """train critic"""
            # train all samples in the buffer, to the extreme
            # self.critic.fit(self.policy, self.buffer, epochs=20, num_samples=self.buffer.size())
            # train some samples minibatches only
            critic_loss_mean, critic_loss_std = self.critic.another_fit_func(self.policy, self.buffer, gradient_steps)

            """calculation of episodic discounted return only needs rewards"""
            mc_returns = np.concatenate([self.discounted_sum(t['scaled_rewards'], self.discount) for t in trajectories])

            """using current batch of samples to update baseline"""
            observes = np.concatenate([t['observes'] for t in trajectories])
            actions = np.concatenate([t['actions'] for t in trajectories])
            value_func_loss = self.value_func.update(observes, mc_returns)

            """compute GAE"""
            for t in trajectories:
                t['values'] = self.value_func.predict(t['observes'])
                # IS it really legitimate to insert 0 at the last obs?
                t['td_residual'] = t['scaled_rewards'] + self.discount * np.append(t['values'][1:], 0) - t['values']
                t['gae'] = self.discounted_sum(t['td_residual'], self.discount * self.lamb)
            advantages = np.concatenate([t['gae'] for t in trajectories])
            """normalize advantage estimates, Crucial step"""
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            """compute control variate"""""
            cv = self.critic.get_contorl_variate(self.policy, observes, actions)
            # cv must not be centered
            # cv = (cv - cv.mean()) / (cv.std() + 1e-6)

            """conservative control variate"""
            eta = [1 if i > 0 else 0 for i in advantages*cv]

            """center learning signal"""
            # check that advantages and CV should be of size E*T
            # eta controls the on-off of control variate
            learning_signal = advantages - eta*cv
            # learning_signal = (learning_signal - learning_signal.mean()) / (learning_signal.std() + 1e-6)

            """controlled taylor eval term"""
            ctrl_taylor = np.concatenate([ [eta[i]*act] for i, act in enumerate(self.critic.get_taylor_eval(self.policy, observes))])

            """policy update"""
            ppo_loss, ddpg_loss, kl, entropy, beta = self.policy.update(observes, actions, learning_signal, ctrl_taylor)

            avg_rewards = np.sum(np.concatenate([t['rewards'] for t in trajectories])) / self.episodes
            avg_timesteps = np.average([len(t['rewards']) for t in trajectories])
            log = {}

            # save training statistics
            log['steps'] = avg_timesteps
            log['rewards'] = avg_rewards
            log['critic_loss'] = critic_loss_mean
            log['policy_ppo_loss'] = ppo_loss
            log['policy_ddpg_loss'] = ddpg_loss
            log['kl'] = kl
            log['entropy'] = entropy
            log['value_func_loss'] = value_func_loss
            log['beta'] = beta

            # display
            print('episode: ', i)
            print('average steps: {0}, average rewards: {1}'.format(log['steps'], log['rewards']))
            for key in ['critic_loss', 'policy_ppo_loss', 'policy_ddpg_loss', 'value_func_loss', 'kl', 'entropy', 'beta']:
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

            # save model weights if stopped early
            if self.killer.kill_now:
                if input('Terminate training (y/[n])? ') == 'y':
                    break
                self.killer.kill_now = False

        self.policy.save(OUTPATH)
        self.value_func.save(OUTPATH)
        self.critic.save(OUTPATH)
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
        """
        Load all Function Approximators plus a Scaler.
        Replaybuffer is not restored though.
        :param load_from: Dir containing saved weights.
        """
        from tensorflow.python.tools import inspect_checkpoint as chkp
        # # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(load_from+'policy/policy.pl', tensor_name='', all_tensors=True, all_tensor_names=True)
        self.policy.load(load_from + 'policy/')
        self.value_func.load(load_from + 'value_func/')
        self.critic.load(load_from+'critic/')
        with open(load_from + "scaler.pkl", 'rb') as file:
            self.scaler = pickle.load(file)

    def demonstrate_agent(self, load_from):
        """
        Simply run the policy without training.
        :param load_from:
        :return:
        """
        self.load_model(load_from)
        while True:
            observes, actons, rewards = self.run_one_episode()
            ep_rewards = np.sum(rewards)
            ep_steps = len(rewards)
            print("Total steps: {0}, total rewards: {1}\n".format(ep_steps, ep_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, help='Number of episodes to run', default=1000)
    parser.add_argument('-d', '--discount', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-k', '--kl_target', type=float, help='KL target', default=0.003)
    parser.add_argument('-l', '--lamb', type=float, help='Lambda for Generalized Advantage Estimation', default=0.98)
    parser.add_argument('-a', '--animate', type=bool, help='Render animation', default=False)
    parser.add_argument('-s', '--show', type=bool, help='Demonstrate a trained agent', default=False)
    parser.add_argument('-r', '--resume', type=bool, help='Resume training', default=False)
    parser.add_argument('--show_dir', type=str, help='The saved model parameters to a trained agent; Must be specified either -s or -r is True')
    parser.add_argument('-m', '--message', type=str, help='Message/identifier for experiments', default="Default")
    args = parser.parse_args()

    if not args.show and not args.resume:   # Training from scratch
        print('Training an agent anew')
        global OUTPATH
        OUTPATH = './results/QPROP/RoboschoolHumanoidFlagrun-v1_' + args.message + '/'  + date_id
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        print("Save location:", OUTPATH)
        del args.message
        del args.show_dir
        expr = Experiment(**vars(args))
        expr.run_expr()
    elif args.show: # Demonstrate a trained agent
        print('Loading a trained agent')
        del args.message
        if args.show_dir is None:
            print('Needs to specify --show_dir when --show is active')
            exit()
        if not args.animate:
            print('Suggest setting --animate to True')
        show_dir = args.show_dir # e.g. "./results/Hopper-v2/offline-PPO/2018-04-06_12_58_36/"
        print('From', show_dir)
        del show_dir
        expr = Experiment(**vars(args))
        expr.demonstrate_agent(show_dir)
    else: # Resume training
        if args.show_dir is None:
            print('Needs to specify --show_dir when --show is active')
            exit()
        show_dir = args.show_dir
        print('Resume training from {}'.format(show_dir))
        global OUTPATH
        OUTPATH = './results/QPROP/RoboschoolHumanoidFlagrun-v1_Resumed/' + date_id
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        print("Save location:", OUTPATH)
        del args.message
        del args.show_dir
        expr = Experiment(**vars(args))
        expr.load_model(show_dir)
        expr.run_expr()
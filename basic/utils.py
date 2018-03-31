import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from basic.tf_pg import Policy
import gym

class l2TargetValueFunc:
    def __init__(self, obs_dim, discount=0.99, lamb=0.8):
        self.obs_dim = obs_dim
        self.discount = discount
        self.lamb = lamb
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'advantages_ph')

            out = tf.layers.dense(self.obs_ph, self.obs_dim, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(1 / self.obs_dim)),
                                  name="valfunc_d1")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)),
                                  name='output')
            # out = tf.Print(out, [out], message='out: ')
            self.out = tf.squeeze(out)  # remove dimensions of size 1 from the shape
            # self.out = tf.Print(out, [out], message='value prediction: ')
            # gradient ascent
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(1e-1)
            self.train = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def update(self, x, y):
        feed_dict = {self.obs_ph: x,
                     self.val_ph: y
                     }

        # add experience replay

        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        # targets_hat = self.predict(x)  # check explained variance prior to update
        # old_exp_var = 1 - np.var(targets - targets_hat) / np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
            self.replay_buffer_x = x
            self.replay_buffer_y = y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
            self.replay_buffer_x = np.concatenate([x, self.replay_buffer_x])
            self.replay_buffer_y = np.concatenate([y, self.replay_buffer_y])
        for e in range(1):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train, self.loss], feed_dict=feed_dict)
        # y_hat = self.predict(x)
        # loss = np.mean(np.square(y_hat - y))  # explained variance after update
        # exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
        # self.sess.run([self.train, self.loss], feed_dict)

    def predict(self, obs):
        feed_dict = {self.obs_ph: obs}
        val = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(val)

    def close_sess(self):
        self.sess.close()

class l2Experiment:

    def __init__(self, env_name, discount=0.99):
        self.env = gym.make(env_name)
        self.obs_dim = 400 #self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.policy = Policy(self.obs_dim, self.act_dim, self.env.action_space)
        self.value_func = l2TargetValueFunc(self.obs_dim)

        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)

        # featurizer!
        observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def featurize_obs(self, obs):
        """
           Returns the featurized representation for a state.
        """
        featurized = None
        try:
            scaled = self.scaler.transform([obs])
            featurized = self.featurizer.transform(scaled)
        except ValueError:
            print('We have a problem with the featurizer!')
        return featurized[0]

    def run_one_epsisode(self):
        obs = self.env.reset()
        obs = self.featurize_obs(obs)
        obs = obs.astype(np.float64).reshape((1, -1))
        rewards = []
        done = False
        step = 0
        while not done:
            # print('samping')
            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)
            # print('action', action)

            obs_new, reward, done, _ = self.env.step(action)
            obs_new = obs_new.astype(np.float64).reshape((-1,))  # major mistake solved here
            obs_new = self.featurize_obs(obs_new)
            obs_new = obs_new.astype(np.float64).reshape((1, -1))

            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            # print('computing advantage')
            target = reward + self.discount * self.value_func.predict(obs_new)
            advantage = target - self.value_func.predict(obs)
            advantage = advantage.astype(np.float64).reshape((1,))
            target = target.astype(np.float64).reshape((1,))

            # print('advantage', advantage)

            self.policy.update(obs, action, advantage)
            self.value_func.update(obs, target)

            obs = obs_new
            step += 0.001

        return rewards

def exp_l2_trace():
    """
    failed; l2 target should really have some MC return instead of bootstraped value estimation.
    :return: disappointment
    """
    env = l2Experiment('MountainCarContinuous-v0')

    steps = []
    undiscounted = []
    # 100 episodes
    for i in range(100):
        # trace vectors are emptied at the beginning of each episode
        env.policy.sess.run(env.policy.trace_zero)
        # env.value_func.replay_buffer_x = None
        # env.value_func.replay_buffer_y = None

        print('episode: ', i)
        rewards = env.run_one_epsisode()
        total_steps = len(rewards)
        print('total steps: {0}, episode_reward: {1}'.format(total_steps, np.sum(rewards)))
        steps.append(total_steps)
        undiscounted.append(np.sum(rewards))

    plt.subplot(121)
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.plot(steps)

    plt.subplot(122)
    plt.xlabel('episode')
    plt.ylabel('undiscounted rewards')
    plt.plot(undiscounted)
    plt.show(0)


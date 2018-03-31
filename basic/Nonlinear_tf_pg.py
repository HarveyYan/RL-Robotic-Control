import tensorflow as tf
import numpy as np
import gym
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt


class NonlinearPolicy:

    def __init__(self, obs_dim, act_dim, action_space, discount=0.99, lamb=0.8):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.discount = discount
        self.lamb = lamb
        self.epochs = 20  # unused when eligibility trace is active
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn_mu()
            self._policy_nn_sigma()
            self._log_prob()
            self._sample()
            self._trace_init()
            self._loss()
            self._trace()
            self._train()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.advantages_ph = tf.placeholder(tf.float32, (1,), 'advantage')  # number of time steps, usually
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')

    def _policy_nn_mu(self):
        units = self.obs_dim * 10
        out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='dense_mu_1')
        out = tf.layers.dense(out, self.act_dim,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_mu')
        self.means = out

    def _policy_nn_sigma(self):
        units = self.obs_dim * 10
        out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='dense_sigma_1')
        out = tf.layers.dense(out, self.act_dim,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_sigma')

        # assuming a diagonal covariance for the multi-variate gaussian distributionlogvar_speed = (10 * hid3_size) // 48
        self.log_vars = out

    def _log_prob(self):
        self.logp = -0.5 * (
                    tf.reduce_sum(self.log_vars) + self.act_dim * tf.log(2 * np.pi))  # probability of a trajectory
        self.logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                          tf.exp(self.log_vars), axis=1)

    def _sample(self):
        self.sample = self.means + tf.exp(self.log_vars / 2.0) * \
                      tf.random_normal(shape=(self.act_dim,))
        self.sample = tf.clip_by_value(self.sample, self.action_space.low[0], self.action_space.high[0])

    def _trace_init(self):
        tvs = tf.trainable_variables()
        self.trace = [(tf.Variable(tf.zeros_like(tv), trainable=False)) for tv in tvs]
        # reset ops
        self.trace_zero = [self.trace[i].assign(tf.zeros_like(tv)) for i, tv in enumerate(tvs)]

    def _loss(self):
        self.loss = -self.logp
        self.loss -= 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                            tf.reduce_sum(self.log_vars))

    def _trace(self):
        self.optimizer = tf.train.AdamOptimizer(1e-2)
        self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
        self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + grad[0]) for i, grad in
                             enumerate(self.grads)]

    def _train(self):
        self.train = self.optimizer.apply_gradients(
            [(self.trace[i] * self.advantages_ph, grad[1]) for i, grad in enumerate(self.grads)])

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def get_sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sample, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages
                     }
        self.sess.run(self.trace_update, feed_dict)
        self.sess.run([self.train, self.loss], feed_dict)

    def close_sess(self):
        self.sess.close()


class NonlinearValueFunc:

    def __init__(self, obs_dim, discount=0.99, lamb=0.8):
        self.obs_dim = obs_dim
        self.epochs = 10
        self.discount = discount
        self.lamb = lamb
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
            self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages_ph')

            units = self.obs_dim * 10
            out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)),
                                  name="valfunc_d1")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)),
                                  name='output')
            self.out = tf.squeeze(out)
            # gradient ascent
            self.loss = -self.out
            # initialize trace
            tvs = tf.trainable_variables()
            self.trace = [(tf.Variable(tf.zeros_like(tv), trainable=False)) for tv in tvs]
            # reset ops
            self.trace_zero = [self.trace[i].assign(tf.zeros_like(tv)) for i, tv in enumerate(tvs)]

            self.optimizer = tf.train.AdamOptimizer(1e-2)
            self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
            self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + grad[0]) for i, grad
                                 in enumerate(self.grads)]

            self.train = self.optimizer.apply_gradients(
                [(self.trace[i] * self.advantages_ph, grad[1]) for i, grad in enumerate(self.grads)])
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def update(self, observes, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.advantages_ph: advantages
                     }

        self.sess.run(self.trace_update, feed_dict)
        self.sess.run([self.train, self.loss], feed_dict)

    def predict(self, obs):
        feed_dict = {self.obs_ph: obs}
        val = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(val)

    def close_sess(self):
        self.sess.close()


class NonlinearExperiment:

    def __init__(self, env_name, discount=0.99):
        self.env = gym.make(env_name)
        self.obs_dim = 401  # add time dimension
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.policy = NonlinearPolicy(self.obs_dim, self.act_dim, self.env.action_space)
        self.value_func = NonlinearValueFunc(self.obs_dim)

        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)

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
        scaled = None
        try:
            scaled = self.scaler.transform([obs])
        except ValueError:
            pass
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def run_one_epsisode(self):
        obs = self.env.reset()
        step = 0
        obs = self.featurize_obs(obs)
        obs = np.append(obs, [step], axis=0)
        obs = obs.astype(np.float64).reshape((1, -1))
        observes, actions, rewards = [], [], []
        done = False
        while not done:
            step += 0.001
            action = self.policy.get_sample(obs).reshape((1, -1)).astype(np.float64)

            obs_new, reward, done, _ = self.env.step(action)
            obs_new = obs_new.astype(np.float64).reshape((-1,))
            obs_new = self.featurize_obs(obs_new)
            obs_new = np.append(obs_new, [step], axis=0)
            obs_new = obs_new.astype(np.float64).reshape((1, -1))

            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)

            advantage = reward + self.discount * self.value_func.predict(obs_new) - self.value_func.predict(obs)
            advantage = advantage.astype(np.float64).reshape((1,))

            self.policy.update(obs, action, advantage)
            self.value_func.update(obs, advantage)

            obs = obs_new
        return observes, actions, rewards


if __name__ == "__main__":
    env = NonlinearExperiment('MountainCarContinuous-v0')
    steps = []
    undiscounted = []
    # 100 episodes
    for i in range(100):
        # trace vectors are emptied at the beginning of each episode
        env.policy.sess.run(env.policy.trace_zero)
        env.value_func.sess.run(env.value_func.trace_zero)

        print('episode: ', i)
        _, _, rewards = env.run_one_epsisode()
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
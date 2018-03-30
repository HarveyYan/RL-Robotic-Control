import tensorflow as tf
import numpy as np
import gym
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Policy:

    def __init__(self, obs_dim, act_dim, action_space, discount=1.0, lamb=0.4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.discount = discount
        self.lamb = lamb
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
        '''
        Add placeholders to the graph
        :return:
        '''
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.advantages_ph = tf.placeholder(tf.float32, (1,), 'advantage')  # number of time steps, usually
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')

    def _policy_nn_mu(self):
        # out = tf.layers.dense(self.obs_ph, units,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / self.obs_dim)),
        #                       name = 'dense_mu_1')
        # out = tf.layers.dense(out, units,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / units)),name='dense_mu_2')
        out = tf.layers.dense(self.obs_ph, self.act_dim, #tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_mu')
        self.means = out # tf.Print(out, [out], message='Mean: ', summarize=100)

    def _policy_nn_sigma(self):
        # out = tf.layers.dense(self.obs_ph, units,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / self.obs_dim)),
        #                       name='dense_sigma_1')
        # out = tf.layers.dense(out, units,
        #                       kernel_initializer=tf.random_normal_initializer(
        #                           stddev=np.sqrt(1 / units)), name='dense_sigma_2')
        out = tf.layers.dense(self.obs_ph, self.act_dim, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_sigma')
        # assuming a diagonal covariance for the multi-variate gaussian distributionlogvar_speed = (10 * hid3_size) // 48
        self.log_vars = out

        # self.log_vars = tf.get_variable('variance', (self.act_dim), tf.float32)
        # self.log_vars = tf.Print(self.log_vars, [self.log_vars], message='log var: ')

    def _log_prob(self):
        # add time dim to log_var?
        self.logp = -0.5 * (tf.reduce_sum(self.log_vars) + self.act_dim*tf.log(2*np.pi))  # probability of a trajectory
        self.logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                          tf.exp(self.log_vars), axis=1)

    def _sample(self):
        self.sample = self.means + tf.exp(self.log_vars / 2.0) * \
                       tf.random_normal(shape=(self.act_dim,))
        # self.sample = tf.clip_by_value(self.sample, self.action_space.low[0], self.action_space.high[0])

    def _trace_init(self):
        """
        Initialize the trace vector; same structure with the gradients
        :return:
        """
        tvs = tf.trainable_variables()
        self.trace = [(tf.Variable(tf.zeros_like(tv), trainable=False)) for tv in tvs]
        self.identity = tf.Variable(1.0, trainable = False, name='identity')
        # reset ops
        self.trace_zero = [self.trace[i].assign(tf.zeros_like(tv)) for i, tv in enumerate(tvs)]
        self.identity_init = [self.identity.assign(1.0)]

    def _loss(self):
        self.loss = -self.logp
        self.loss -= 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))
        self.loss *= self.advantages_ph
        # self.loss -= 1e-1 * self.normal_dist.entropy()

    def _trace(self):
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
        self.identity_update = [self.identity.assign(self.identity*self.discount)]
        # self.identity = tf.Print(self.identity, [self.identity], message ='identity: ')
        self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + self.identity * grad[0]) for i, grad in
                             enumerate(self.grads)]

    def _train(self):
        self.train = self.optimizer.apply_gradients(
            [(self.trace[i], grad[1]) for i, grad in enumerate(self.grads)])

    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def get_sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        action = self.sess.run(self.sample, feed_dict=feed_dict)
        return action

    def update(self, observes, actions, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages
                     }

        # self.sess.run(self.loss, feed_dict)
        # self.sess.run(self.grads, feed_dict)
        self.sess.run(self.trace_update, feed_dict)
        # self.sess.run(self.identity_update)
        self.sess.run([self.train,self.loss], feed_dict)

    def close_sess(self):
        self.sess.close()

class ValueFunc:
    def __init__(self, obs_dim, discount=1.0, lamb=0.4):
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
            # out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
            #                       kernel_initializer=tf.random_normal_initializer(
            #                         stddev=np.sqrt(1 / self.obs_dim)),
            #                       name="valfunc_d1")
            # out = tf.layers.dense(out, units,
            #                       kernel_initializer=tf.random_normal_initializer(
            #                           stddev=np.sqrt(1 / units)),
            #                       name="valfunc_d2")
            out = tf.layers.dense(self.obs_ph, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)),
                                  name='output')
            # out = tf.Print(out, [out], message='out: ')
            self.out = tf.squeeze(out)  # remove dimensions of size 1 from the shape
            # self.out = tf.Print(out, [out], message='value prediction: ')
            # gradient ascent
            self.loss = -self.out * self.advantages_ph
            # initialize trace
            tvs = tf.trainable_variables()
            self.trace = [(tf.Variable(tf.zeros_like(tv), trainable=False)) for tv in tvs]
            self.identity = tf.Variable(1.0, trainable=False, name='identity')
            # reset ops
            self.trace_zero = [self.trace[i].assign(tf.zeros_like(tv)) for i, tv in enumerate(tvs)]
            self.identity_init = [self.identity.assign(1.0)]

            self.optimizer = tf.train.AdamOptimizer(1e-3)
            self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
            self.identity_update = [self.identity.assign(self.identity*self.discount)]
            # self.identity = tf.Print(self.identity, [self.identity], message='identity: ')
            self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + self.identity * grad[0]) for i, grad
                                 in enumerate(self.grads)]

            self.train = self.optimizer.apply_gradients(
                [(self.trace[i], grad[1]) for i, grad in enumerate(self.grads)])
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def update(self, observes, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.advantages_ph: advantages
                     }

        self.sess.run(self.trace_update, feed_dict)
        # self.sess.run(self.identity_update)
        _, loss = self.sess.run([self.train, self.loss], feed_dict)
        return loss

    def predict(self, obs):
        feed_dict = {self.obs_ph: obs}
        val = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(val)

    def close_sess(self):
        self.sess.close()

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

            # out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
            #                       kernel_initializer=tf.random_normal_initializer(
            #                         stddev=np.sqrt(1 / self.obs_dim)),
            #                       name="valfunc_d1")
            # out = tf.layers.dense(out, units,
            #                       kernel_initializer=tf.random_normal_initializer(
            #                           stddev=np.sqrt(1 / units)),
            #                       name="valfunc_d2")
            out = tf.layers.dense(self.obs_ph, 1,
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

class Experiment:

    def __init__(self, env_name, discount=1.0):
        self.env = gym.make(env_name)
        self.obs_dim = 400 #self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.policy = Policy(self.obs_dim, self.act_dim, self.env.action_space)
        self.value_func = ValueFunc(self.obs_dim)

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
            # self.env.render()

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
            advantage = reward + self.discount * self.value_func.predict(obs_new) - self.value_func.predict(obs)
            advantage = advantage.astype(np.float64).reshape((1,))

            # print('advantage', advantage)

            # print('policy update')
            loss = self.policy.update(obs, action, advantage)
            # print('value function update')
            self.value_func.update(obs, advantage)

            obs = obs_new
            step += 0.001

        return rewards

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
            # self.env.render()

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

def exp_whole_trace():
    env = Experiment('MountainCarContinuous-v0')

    steps = []
    undiscounted = []
    # 100 episodes
    for i in range(1000):
        # trace vectors are emptied at the beginning of each episode
        env.policy.sess.run(env.policy.trace_zero)
        env.value_func.sess.run(env.value_func.trace_zero)
        env.policy.sess.run(env.policy.identity_init)
        env.value_func.sess.run(env.value_func.identity_init)

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
    plt.show()

def exp_l2_trace():
    """
    failed; l2 target should really be MC return
    :return:
    """
    env = l2Experiment('MountainCarContinuous-v0')
    # env.policy.lamb = 0

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
    plt.show()

# network too big will explode!!
if __name__ == "__main__":
    exp_whole_trace()

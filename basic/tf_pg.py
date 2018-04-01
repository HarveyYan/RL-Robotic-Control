import tensorflow as tf
import numpy as np
import gym
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
import datetime
import os

OUTPATH = './Results/cmc/' + str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_') + '/'

tf.app.flags.DEFINE_string('checkpoint_dir', OUTPATH,
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
FLAGS = tf.app.flags.FLAGS


class Policy:

    def __init__(self, obs_dim, act_dim, action_space, discount=1.0, lamb=0.8):
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
            self.saver = tf.train.Saver()
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
        out = tf.layers.dense(self.obs_ph, self.obs_dim, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='dense_mu_1')
        out = tf.layers.dense(out, self.act_dim,  # tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_mu')
        self.means = out
        # tf.Print(out, [out], message='Mean: ', summarize=100)

    def _policy_nn_sigma(self):
        out = tf.layers.dense(self.obs_ph, self.obs_dim, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='dense_sigma_1')
        out = tf.layers.dense(out, self.act_dim,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)),
                              name='output_sigma')

        # assuming a diagonal covariance for the multi-variate gaussian distributionl
        self.log_vars = out

    def _log_prob(self):
        '''
        probability of a trajectory, or a single observation and action.
        :return:
        '''
        self.logp = -0.5 * (
                    tf.reduce_sum(self.log_vars) + self.act_dim * tf.log(2 * np.pi))  # probability of a trajectory
        self.logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                          tf.exp(self.log_vars), axis=1)

    def _sample(self):
        '''
        sample from a parameterized gaussian
        :return:
        '''
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
        self.identity = tf.Variable(1.0, trainable=False, name='identity')
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
        self.optimizer = tf.train.AdamOptimizer(1e-5)
        self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
        self.identity_update = [self.identity.assign(self.identity * self.discount)]
        # self.identity = tf.Print(self.identity, [self.identity], message ='identity: ')
        self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + self.identity * grad[0])
                             for i, grad in
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
        self.sess.run([self.train, self.loss], feed_dict)

    def initialize_trace(self):
        self.sess.run(self.trace_zero)
        self.sess.run(self.identity_init)

    def save(self):
        if not os.path.exists(OUTPATH + 'policy'):
            os.makedirs(OUTPATH + 'policy')
        self.saver.save(self.sess, OUTPATH + 'policy/policy.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)
        pass

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
            self.identity_update = [self.identity.assign(self.identity * self.discount)]
            # self.identity = tf.Print(self.identity, [self.identity], message='identity: ')
            self.trace_update = [
                self.trace[i].assign(self.discount * self.lamb * self.trace[i] + self.identity * grad[0]) for i, grad
                in enumerate(self.grads)]

            self.train = self.optimizer.apply_gradients(
                [(self.trace[i], grad[1]) for i, grad in enumerate(self.grads)])
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
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

    def initialize_trace(self):
        self.sess.run(self.trace_zero)
        self.sess.run(self.identity_init)

    def save(self):
        # self.builder = tf.saved_model.builder.SavedModelBuilder(OUTPATH + 'value-func/')
        if not os.path.exists(OUTPATH + 'value-func'):
            os.makedirs(OUTPATH + 'value-func')
        self.saver.save(self.sess, OUTPATH + 'value-func/value-func.pl')
        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        # self.builder.add_meta_graph_and_variables(self.sess,
        #                                           [tf.saved_model.tag_constants.SERVING],
        #                                           signature_def_map={
        #                                               "estimating":tf.saved_model.signature_def_utils.build_signature_def(
        #                                                 inputs={"obs": tf.saved_model.utils.build_tensor_info(self.obs_ph)},
        #                                                 outputs= {"estimate":tf.saved_model.utils.build_tensor_info(self.out)},
        #                                                   method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        #                                               )
        #                                           }, legacy_init_op = legacy_init_op)

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)

    def close_sess(self):
        self.sess.close()


class Experiment:

    def __init__(self, env_name, discount=1.0):
        self.env = gym.make(env_name)
        gym.spaces.seed(1234)   # same observation space samples everytime
        self.obs_dim = 400  # self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.discount = discount
        self.policy = Policy(self.obs_dim, self.act_dim, self.env.action_space)
        self.value_func = ValueFunc(self.obs_dim)

        print('observation dimension:', self.obs_dim)
        print('action dimension:', self.act_dim)

        # featurizer!
        observation_examples = np.array([self.env.observation_space.sample() for x in range(100000)])
        # print(observation_examples)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # set seed to reproduce experiments with saved weights
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100, random_state=1234)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100, random_state=1234)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100, random_state=1234)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100, random_state=1234))
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

    def run_one_episode(self, save=True, train=True, animate=False):
        # initializing trace
        self.policy.initialize_trace()
        self.value_func.initialize_trace()

        obs = self.env.reset()
        obs = self.featurize_obs(obs)
        obs = obs.astype(np.float64).reshape((1, -1))
        rewards = []
        done = False
        step = 0
        while not done:
            if animate:
                self.env.render()
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

            if train:
                advantage = reward + self.discount * self.value_func.predict(obs_new) - self.value_func.predict(obs)
                advantage = advantage.astype(np.float64).reshape((1,))

                loss = self.policy.update(obs, action, advantage)
                self.value_func.update(obs, advantage)

            obs = obs_new
            step += 0.001

        if save:
            self.policy.save()
            self.value_func.save()

        return rewards

    def load_model(self, load_from):
        from tensorflow.python.tools import inspect_checkpoint as chkp

        # # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(load_from+'policy/policy.pl', tensor_name='', all_tensors=True, all_tensor_names=True)
        self.policy.load(load_from + 'policy/policy.pl')
        self.value_func.load(load_from + 'value-func/value-func.pl')


def load_from_experiment(path='./Results/cmc/2018-03-31_12_18_20/'):
    env = Experiment('MountainCarContinuous-v0')

    print('Loading FA weights from', path)
    env.load_model(path)

    steps = []
    undiscounted = []
    # 10 episodes
    for i in range(50):
        # trace vectors are emptied at the beginning of each episode
        print('episode: ', i)
        rewards = env.run_one_episode(save=False, train=False, animate=True)
        total_steps = len(rewards)
        print('total steps: {0}, episode_reward: {1}'.format(total_steps, np.sum(rewards)))
        steps.append(total_steps)
        undiscounted.append(np.sum(rewards))


def exp_whole_trace():
    env = Experiment('MountainCarContinuous-v0')

    steps = []
    undiscounted = []
    # 10 episodes
    for i in range(10):
        # trace vectors are emptied at the beginning of each episode
        print('episode: ', i)
        # if i < 5:
        rewards = env.run_one_episode(save=i == 9, train=True)
        # else:
        #     rewards = env.run_one_episode(save=False, train=False)
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
    plt.savefig(OUTPATH + 'train.png')


if __name__ == "__main__":
    # exp_whole_trace()
    load_from_experiment()

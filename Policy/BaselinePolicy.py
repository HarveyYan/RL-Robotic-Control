import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
gpu_options.allow_growth = True

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
        units = 10 * self.obs_dim
        out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
                              kernel_initializer=tf.zeros_initializer(),
                              name='dense_mu_1')
        out = tf.layers.dense(out, self.act_dim,  # tf.tanh,
                              kernel_initializer=tf.zeros_initializer(),
                              name='output_mu')
        self.means = out# tf.Print(out, [out], message='Mean: ', summarize=100)

    def _policy_nn_sigma(self):
        units = self.obs_dim * 10
        out = tf.layers.dense(self.obs_ph, units, tf.nn.relu,
                              kernel_initializer=tf.zeros_initializer(),
                              name='dense_sigma_1')
        out = tf.layers.dense(out, self.act_dim,
                              kernel_initializer=tf.zeros_initializer(),
                              name='output_sigma')
        # assuming a diagonal covariance for the multi-variate gaussian distribution
        self.log_vars = out #tf.Print(out, [out], message='log vars: ', summarize=100)

    def _log_prob(self):
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
        self.loss = -self.logp*self.advantages_ph
        self.loss -= 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))
        # self.loss -= 1e-1 * self.normal_dist.entropy()

    def _trace(self):
        self.optimizer = tf.train.AdamOptimizer(1e-5)
        self.grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
        self.identity_update = [self.identity.assign(self.identity*self.discount)]
        # self.identity = tf.Print(self.identity, [self.identity], message ='identity: ')
        self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + self.identity * grad[0]) for i, grad in
                             enumerate(self.grads)]

    def _train(self):
        self.train = self.optimizer.apply_gradients(
            [(self.trace[i], grad[1]) for i, grad in enumerate(self.grads)])

    def _init_session(self):
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def get_sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sample, feed_dict=feed_dict)

    def init_trace(self):
        self.sess.run(self.trace_zero)

    def update(self, observes, actions, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages
                     }
        self.sess.run(self.trace_update, feed_dict)
        # self.sess.run(self.identity_update)
        self.sess.run([self.train,self.loss], feed_dict)

    def save(self, saveto):
        if not os.path.exists(saveto + 'policy'):
            os.makedirs(saveto + 'policy')
        self.saver.save(self.sess, saveto + 'policy/policy.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)

    def close_sess(self):
        self.sess.close()
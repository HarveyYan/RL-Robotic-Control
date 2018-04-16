"""
without eligibility trace, this one is almost identical to the reference policy implementation.
"""

import tensorflow as tf
import tensorflow.contrib.layers
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True

class NoTracePolicy:

    def __init__(self, obs_dim, act_dim, action_space, kl_target, epochs=1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.kl_target = kl_target
        self.epochs = epochs
        self.eta = 50 # hinge loss multiplier, between actual kl and kl target
        self.beta = 1.0 # kl penalty term multiplier
        self.lr = 1e-4
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn_mu()
            # self._policy_nn_sigma()
            self._log_prob()
            self._kl_and_entropy()
            self._sample()
            self._loss()
            self._train()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        '''
        Add placeholders to the graph
        :return:
        '''
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantage')  # number of time steps, usually

        self.means_old_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
        self.logvars_old_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_logvars')

        self.beta_ph = tf.placeholder(tf.float32, (), 'kl_penalty_multiplier')
        self.eta_ph = tf.placeholder(tf.float32, (), 'hinge_penalty_multiplier')
        self.lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')

    def _policy_nn_mu(self):
        hid1_size = self.obs_dim * 10  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h3")
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name="means")

        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0

    def _policy_nn_sigma(self):
        # units_layer_1 = 10 * self.obs_dim
        # units_layer_2 = int(np.sqrt(units_layer_1 * self.act_dim))  # geometirc mean of first and last layers
        # out = tf.layers.dense(self.obs_ph, units_layer_1, tf.nn.relu,
        #                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                       name='dense_sigma_1')
        # out = tf.layers.dense(out, units_layer_2, tf.nn.relu,
        #                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                       name='dense_sigma_2')
        # out = tf.layers.dense(out, self.act_dim,
        #                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                       name='output_sigma')
        # assuming a diagonal covariance for the multi-variate gaussian distribution
        logvar_speed = 6
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0
        # self.log_vars = out #tf.Print(out, [out], message='log vars: ', summarize=100)

    def _log_prob(self):
        logp = -0.5 * tf.reduce_sum(self.log_vars)  # probability of a trajectory
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                          tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.logvars_old_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means_old_ph) /
                                         tf.exp(self.logvars_old_ph), axis=1)
        self.logp_old = logp_old

    def _kl_and_entropy(self):
        """
        Taken directly from Patrick Coady's code. Validity verified.
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.logvars_old_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.logvars_old_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.means_old_ph) /
                                                     tf.exp(self.log_vars), axis=1) - self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """
        Reparametrization trick.
        :return: 
        """
        self.sample = (self.means + tf.exp(self.log_vars / 2.0) *
                       tf.random_normal(shape=(self.act_dim,)))
        # self.sample = tf.clip_by_value(self.sample, self.action_space.low[0], self.action_space.high[0])

    def _loss(self):
        """
        Four loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2
            4) Entropy for encouraging exploration
        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        loss = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old)) # p/p_old
        loss += tf.reduce_mean(self.beta_ph * self.kl)
        loss += self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_target))
        # loss -= self.entropy # encouraged, needs multiplier?
        self.loss = loss

    def _train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train = self.optimizer.minimize(self.loss)

    def _init_session(self):
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def get_sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sample, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """
        Update policy based on observations, actions and advantages
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        # TODO, check validity of means_old and log_vars
        means_old, logvars_old = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.logvars_old_ph] = logvars_old
        feed_dict[self.means_old_ph] = means_old

        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_target * 4:  # early stopping if D_KL diverges badly
                break
        # self.sess.run(self.train, feed_dict)
        # loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)

        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_target * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_target / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        # print(self.beta)
        return loss, kl, entropy, self.beta

    def save(self, saveto):
        if not os.path.exists(saveto + 'policy'):
            os.makedirs(saveto + 'policy')
        self.saver.save(self.sess, saveto + 'policy/policy.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)

    def close_sess(self):
        self.sess.close()
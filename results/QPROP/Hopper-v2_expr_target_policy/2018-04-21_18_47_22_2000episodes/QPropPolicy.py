"""
without eligibility trace, this one is almost identical to the referenced policy implementation.
"""

import tensorflow as tf
import tensorflow.contrib.layers
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True

class QPropPolicy:

    def __init__(self, obs_dim, act_dim, action_space, kl_target, epochs=1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space = action_space
        self.kl_target = kl_target
        self.epochs = epochs
        self.eta = 50 # hinge loss multiplier, between actual kl and kl target
        self.beta = 1.0 # kl penalty term multiplier
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.tao = 1e-3

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._log_prob()
            self._kl_and_entropy()
            self._sample()
            self._loss()
            self._train()
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        # build target policy, which is deterministic for DDPG updates
        self.target_g = tf.Graph()
        with self.target_g.as_default():
            self.target_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            hid1_size = self.obs_dim * 10  # 10 empirically determined
            hid3_size = self.act_dim * 10  # 10 empirically determined
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # 3 hidden layers with relu activations
            out = tf.layers.dense(self.target_obs_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h3")
            self.target_means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name="means")
            self.target_saver = tf.train.Saver()

    def _placeholders(self):
        '''
        Add placeholders to the graph
        :return:
        '''
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.learning_signal_ph = tf.placeholder(tf.float32, (None,), 'learning_signal')  # number of time steps, usually
        self.ctrl_taylor_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'ctrl_taylor')  # number of time steps, usually

        self.means_old_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
        self.logvars_old_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_logvars')

        self.beta_ph = tf.placeholder(tf.float32, (), 'kl_penalty_multiplier')
        self.eta_ph = tf.placeholder(tf.float32, (), 'hinge_penalty_multiplier')
        self.lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')

    def _policy_nn(self):
        """
        Local mean and global diagonal covariance.
        :return:
        """
        hid1_size = self.obs_dim * 10  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
        # 3 hidden layers with relu activations
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
        # TODO, check if two losses are equal
        """
        Four loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2
            4) Entropy for encouraging exploration
        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        """PPO loss definition"""
        self.ppo_loss = -tf.reduce_mean(self.learning_signal_ph * tf.exp(self.logp - self.logp_old))
        self.ppo_loss += tf.reduce_mean(self.beta_ph * self.kl)
        self.ppo_loss += self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_target))
        """DDPG loss definition"""
        # ctrl_taylor_ph is of shape (#samples, act_dim), means is of shape (#samples, act_dim)
        # self.ddpg_loss = -tf.reduce_mean(tf.diag_part(tf.matmul(self.ctrl_taylor_ph, self.means, transpose_b=True)))
        self.ddpg_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.ctrl_taylor_ph, self.means), axis=1))

    def _train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train = self.optimizer.minimize(self.ppo_loss+self.ddpg_loss)

    def _init_session(self):
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

        self.target_sess = tf.Session(graph=self.target_g, config=tf.ConfigProto(gpu_options=gpu_options))

        # initialize target policy to the real policy right here
        with self.g.as_default():
            policy_tvs = tf.trainable_variables()
        with self.target_g.as_default():
            target_tvs = tf.trainable_variables()
            self.target_sess.run([tar_t.assign(policy_tvs[i].eval(session=self.sess)) for i, tar_t in enumerate(target_tvs)])

    def get_sample(self, obs):
        """
        Sample an action from the stochastic policy.
        :param obs:
        :return:
        """
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sample, feed_dict=feed_dict)

    def mean(self, obs):
        """
        Expected action from the determinstic target policy.
        :param obs:
        :return:
        """
        feed_dict = {self.target_obs_ph: obs}
        return self.target_sess.run(self.target_means, feed_dict=feed_dict)

    def update(self, observes, actions, learning_signal, ctrl_taylor):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.learning_signal_ph: learning_signal,
                     self.ctrl_taylor_ph: ctrl_taylor,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        # Necessity of conservative policy update
        means_old, logvars_old = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.logvars_old_ph] = logvars_old
        feed_dict[self.means_old_ph] = means_old

        ppo_loss, ddpg_loss, kl, entropy = 0, 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train, feed_dict)
            ppo_loss, ddpg_loss, kl, entropy = self.sess.run([self.ppo_loss, self.ddpg_loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_target * 4:  # early stopping if D_KL diverges badly
                break

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

        # update target policy
        with self.g.as_default():
            policy_tvs = tf.trainable_variables()
        with self.target_g.as_default():
            target_tvs = tf.trainable_variables()
            self.target_sess.run([tar_t.assign(self.tao * policy_tvs[i].eval(session=self.sess) +
                                               (1-self.tao) * tar_t.eval(session=self.target_sess)) for i, tar_t in enumerate(target_tvs)])

        # return ppo_loss, ddpg_loss, kl, entropy, self.beta
        return ppo_loss, ddpg_loss, kl, entropy, self.beta


    def save(self, saveto):
        if not os.path.exists(saveto + 'policy'):
            os.makedirs(saveto + 'policy')
        self.saver.save(self.sess, saveto + 'policy/policy.pl')
        self.target_saver.save(self.target_sess, saveto + 'policy/target_policy.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from+'policy.pl')
        self.target_saver.restore(self.target_sess, load_from+'target_policy.pl')

    def close_sess(self):
        self.sess.close()
        self.target_sess.close()
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True

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

            units_layer_1 = 10 * self.obs_dim
            units_layer_2 = int(np.sqrt(units_layer_1 * 1))  # geometirc mean of first and last layers
            out = tf.layers.dense(self.obs_ph, units_layer_1, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="valfunc_d1")
            out = tf.layers.dense(out, units_layer_2, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="valfunc_d2")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name='output')

            # out = tf.Print(out, [out], message='out: ')
            self.out = tf.squeeze(out)  # remove dimensions of size 1 from the shape
            # self.out = tf.Print(out, [out], message='value prediction: ')
            # gradient ascent
            self.loss = -self.out
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
            self.trace_update = [self.trace[i].assign(self.discount * self.lamb * self.trace[i] + grad[0]) for i, grad
                                 in enumerate(self.grads)]

            self.train = self.optimizer.apply_gradients(
                [(self.trace[i]*self.advantages_ph, grad[1]) for i, grad in enumerate(self.grads)])
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def update(self, observes, advantages):
        feed_dict = {self.obs_ph: observes,
                     self.advantages_ph: advantages
                     }

        self.sess.run(self.trace_update, feed_dict)
        # in Barto&Sutton's book they have an identity scaler for the trace update, which is not benevolent however.
        # self.sess.run(self.identity_update)
        _ , loss = self.sess.run([self.train, self.loss], feed_dict)
        return loss

    def init_trace(self):
        self.sess.run(self.trace_zero)

    def predict(self, obs):
        feed_dict = {self.obs_ph: obs}
        val = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(val)

    def save(self, saveto):
        # self.builder = tf.saved_model.builder.SavedModelBuilder(OUTPATH + 'value-func/')
        if not os.path.exists(saveto + 'value-func'):
            os.makedirs(saveto + 'value-func')
        self.saver.save(self.sess, saveto + 'value-func/value-func.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)

    def close_sess(self):
        self.sess.close()
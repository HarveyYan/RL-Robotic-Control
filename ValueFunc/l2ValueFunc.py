# we need really accurate value function otherwise bias would kill us

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
gpu_options.allow_growth = True

class l2TargetValueFunc:
    def __init__(self, obs_dim, epochs=10):
        self.obs_dim = obs_dim
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.epochs = epochs
        self._build_graph()
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'advantages_ph')

            hid1_size = self.obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), name='output')
            self.out = tf.squeeze(out)  # remove dimensions of size 1 from the shape
            # self.out = tf.Print(out, [out], message='value prediction: ')
            # gradient ascent
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.saver = tf.train.Saver()
            self.train = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()

    def update(self, x, y):
        """
        Using mini-batch gradient descent is imperative to the model training speed.
        Experience replay and shuffling are also complementary.
        :param x:
        :param y:
        :return:
        """

        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches

        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])

        if self.replay_buffer_x is None or self.replay_buffer_x.shape[0] > 1000:
            self.replay_buffer_x = x
            self.replay_buffer_y = y
        else:
            self.replay_buffer_x = np.concatenate([x, self.replay_buffer_x])
            self.replay_buffer_y = np.concatenate([y, self.replay_buffer_y])

        for e in range(self.epochs):
            # Interesting but not very necessary.
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train, self.loss], feed_dict=feed_dict)

        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))
        return loss

    def save(self, saveto):
        if not os.path.exists(saveto + 'value_func'):
            os.makedirs(saveto + 'value_func')
        self.saver.save(self.sess, saveto + 'value_func/value_func.pl')

    def load(self, load_from):
        self.saver.restore(self.sess, load_from)

    def predict(self, obs):
        feed_dict = {self.obs_ph: obs}
        val = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(val)

    def close_sess(self):
        self.sess.close()
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle

class DeterministicCritic:
    """
    A Q state-value function, trained with off-policy samples from a replay buffer.

    """

    def __init__(self, obs_dim, act_dim, discount, saveto):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.saveto = saveto
        self.discount = discount

        graph, init, self.critic_saver= self._build_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.critic_sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.critic_sess.run(init)

        target_graph, _, self.target_saver = self._build_graph()
        self.target_sess = tf.Session(graph=target_graph)

        self.init_target = True # target network needs to be initialized to Q critic the first time
        self.tao = 1e-3 # mixture parameter, as in the paper DDPG Sec. 7 Experiments details

    def _build_graph(self):
        g = tf.Graph()
        with g.as_default():
            obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
            act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act_ph')
            val_tar_ph = tf.placeholder(tf.float32, (None,), 'val_tar_ph')

            """actions are included until later layers."""

            # 400 and 300 as mentioned in DDPG paper
            hid1_size = 40 #400
            hid2_size = 40 #300
            lr = 1e-3
            # 3 hidden layers with relu activations
            out = tf.layers.dense(obs_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name="h1")
            out = tf.concat([out, act_ph], 1, name="add_act")
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name="h2")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name='output')
            value = tf.squeeze(out, name="value")  # remove dimensions of size 1 from the shape

            # gradient ascent
            loss = tf.reduce_mean(tf.square(value - val_tar_ph), name='loss')  # squared loss
            optimizer = tf.train.AdamOptimizer(lr)
            saver = tf.train.Saver(name='saver')
            train = optimizer.minimize(loss, name='train')
            init = tf.global_variables_initializer()
        return g, init, saver

    # def predict(self, obs, act):
    #     feed_dict = {self.obs_ph: obs, self.act_ph: act}
    #     val = self.sess.run(self.value, feed_dict=feed_dict)
    #     return np.squeeze(val)

    def fit(self, policy, buffer, epochs, num_samples, batch_size=256):
        """
        DDPG training style, fitted with off-policy TD learning as in 'Lillicrap et al., 2016'.
        :param policy: where we can expected action for each observation
        :param buffer: where samples come from
        :param epochs: once we get all the samples, we would use mini-batch training on the critic.
        :param num_samples: Normally would be in the size of Episodes*Time_steps
        :param batch_size: minibatch size
        :return:
        """
        if self.init_target: # initialize target network
            if not os.path.exists(self.saveto + 'critic'):
                os.makedirs(self.saveto + 'critic')
            self.critic_saver.save(self.critic_sess, self.saveto + 'critic/critic.pl')
            self.target_saver.restore(self.target_sess, self.saveto + 'critic/critic.pl')
            self.init_target = False

        """processing training data"""
        # [0]: obs, [1]: act, [2]: rewards, [3]: next_obs
        samples = buffer.sample(num_samples)
        with self.target_sess.as_default():
            graph = self.target_sess.graph
            X = np.array([np.concatenate([obs, act]) for obs, act in zip(*(samples[:2]))])
            y = samples[2] + self.discount * graph.get_tensor_by_name("value:0").eval(feed_dict={
                graph.get_tensor_by_name("obs_ph:0"): samples[3],
                graph.get_tensor_by_name("act_ph:0"): policy.mean(samples[3])
            })

        """training...the critic network"""
        losses = []
        num_batches = max(X.shape[0] // batch_size, 1)
        batch_size = X.shape[0] // num_batches
        with self.critic_sess.as_default():
            graph = self.critic_sess.graph
            for e in range(epochs):
                # Interesting but not very necessary.
                x_train, y_train = shuffle(X, y)
                for j in range(num_batches):
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    feed_dict = {graph.get_tensor_by_name("obs_ph:0"): x_train[start:end, 0:self.obs_dim],
                                 graph.get_tensor_by_name("act_ph:0"): x_train[start:end, self.obs_dim:],
                                 graph.get_tensor_by_name("val_tar_ph:0"): y_train[start:end]}
                    graph.get_operation_by_name("train").run(feed_dict=feed_dict)
                    losses.append(graph.get_tensor_by_name("loss:0").eval(feed_dict=feed_dict))
            losses = np.array(losses)
            loss_mean, loss_std = losses.mean(), losses.std()


        """update target network"""
        with self.critic_sess.graph.as_default():
            critic_tvs = tf.trainable_variables()
        with self.target_sess.graph.as_default():
            target_tvs = tf.trainable_variables()
            self.target_sess.run([tar_t.assign(self.tao * critic_tvs[i].eval(session=self.critic_sess) + (1-self.tao) * tar_t.eval(session=self.target_sess)) for i, tar_t in enumerate(target_tvs)])

        return loss_mean, loss_std

    def another_fit_func(self, policy, buffer, gradient_steps, mini_batch_size=256):

        if self.init_target: # initialize target network
            if not os.path.exists(self.saveto + 'critic'):
                os.makedirs(self.saveto + 'critic')
            self.critic_saver.save(self.critic_sess, self.saveto + 'critic/critic.pl')
            self.target_saver.restore(self.target_sess, self.saveto + 'critic/critic.pl')
            self.init_target = False

        losses = []
        for i in range(gradient_steps):

            """process a minibatch of training data"""
            # [0]: obs, [1]: act, [2]: rewards, [3]: next_obs
            samples = buffer.sample(mini_batch_size)
            with self.target_sess.as_default():
                graph = self.target_sess.graph
                X = np.array([np.concatenate([obs, act]) for obs, act in zip(*(samples[:2]))])
                y = samples[2] + self.discount * graph.get_tensor_by_name("value:0").eval(feed_dict={
                    graph.get_tensor_by_name("obs_ph:0"): samples[3],
                    graph.get_tensor_by_name("act_ph:0"): policy.mean(samples[3])
                })

            """training...the critic network"""
            with self.critic_sess.as_default():
                graph = self.critic_sess.graph
                x_train, y_train = shuffle(X, y)
                # for _ in range(5):
                feed_dict = {graph.get_tensor_by_name("obs_ph:0"): x_train[:, 0:self.obs_dim],
                             graph.get_tensor_by_name("act_ph:0"): x_train[:, self.obs_dim:],
                             graph.get_tensor_by_name("val_tar_ph:0"): y_train}
                graph.get_operation_by_name("train").run(feed_dict=feed_dict)
                losses.append(graph.get_tensor_by_name("loss:0").eval(feed_dict=feed_dict))

        losses = np.array(losses)
        loss_mean, loss_std = losses.mean(), losses.std()


        """update target network"""
        with self.critic_sess.graph.as_default():
            critic_tvs = tf.trainable_variables()
        with self.target_sess.graph.as_default():
            target_tvs = tf.trainable_variables()
            self.target_sess.run([tar_t.assign(self.tao * critic_tvs[i].eval(session=self.critic_sess) +
                                               (1-self.tao) * tar_t.eval(session=self.target_sess)) for i, tar_t in enumerate(target_tvs)])

        return loss_mean, loss_std

    def get_contorl_variate(self, policy, observes, actions):
        """
        observes should be in the shape (#samples,obs_dim)
        :param policy:
        :param observes:
        :return:
        """
        expected_actions = policy.mean(observes)
        term_mul = actions - expected_actions
        with self.critic_sess.as_default():
            graph = self.critic_sess.graph
            grads = tf.gradients(graph.get_tensor_by_name("value:0"), graph.get_tensor_by_name("act_ph:0"))[0].eval(feed_dict={
                graph.get_tensor_by_name('obs_ph:0'): observes,
                graph.get_tensor_by_name('act_ph:0'): expected_actions
            })
            # grads are of shape (#samples, act_dim)
        # cv = np.diag(np.matmul(grads, term_mul.T))
        cv = np.sum(np.multiply(grads, term_mul), axis=1)
        return cv

    def get_taylor_eval(self, policy, observes):
        expected_actions = policy.mean(observes)
        with self.critic_sess.as_default():
            graph = self.critic_sess.graph
            grads = tf.gradients(graph.get_tensor_by_name("value:0"), graph.get_tensor_by_name("act_ph:0"))[0].eval(feed_dict={
                graph.get_tensor_by_name('obs_ph:0'): observes,
                graph.get_tensor_by_name('act_ph:0'): expected_actions
            })
        return grads


if __name__ == "__main__":
    critic = DeterministicCritic(1,2,0.8,'./tmp/')
    # critic.fit(None, None, None)
    with critic.critic_sess.as_default():
        graph = critic.critic_sess.graph
        grad = tf.gradients(graph.get_tensor_by_name("value:0"),
                     graph.get_tensor_by_name("act_ph:0"))[0].eval(feed_dict={
            graph.get_tensor_by_name('act_ph:0'): [[1, 2]],
            graph.get_tensor_by_name('obs_ph:0'): [[1]]
        })
        print(grad)
        print(grad.shape)
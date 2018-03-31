import tensorflow as tf
import numpy as np

#date generation
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

#linear model
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

#minimize variance
loss = tf.reduce_sum(tf.square(y - y_data)) #why I cannot use sum here
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#initialization
init = tf.global_variables_initializer()

#graph initialization
sess = tf.Session()
sess.run(init)

#train network
for step in range(201):
    sess.run(train)
    print(step, sess.run(W), sess.run(b), sess.run(loss))
#if step % 20 == 0:
import tensorflow as tf

def _build_graph():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=(2), name='input_x')
        y = tf.placeholder(tf.float32, shape=(2), name='input_y')
        x_new = tf.expand_dims(x, 1)
        y_new = tf.expand_dims(y, 1)
        # x_new = tf.Print(x_new, [x_new], message='X: ', name='print_raw_x')
        # y_new = tf.Print(y_new, [y_new], message='Y: ', name='print_raw_y')
        '''
        returns the shape of a tensor
        '''
        # shape_x = tf.shape(x_new)
        # shape_y = tf.shape(y_new)
        # print_x = tf.Print(shape_x, [shape_x], message='Shape of x: ', name='print_x')
        # print_y = tf.Print(shape_y, [shape_y], message='Shape of y: ', name='print_y')

        node1 = tf.Variable([[3.0], [2.0]],name='node1')
        node2 = tf.Variable([[4.0], [5.0]],name='node2')

        node1 = tf.add(node1, x_new)
        # node1 = tf.Print(node1, [node1], message='node1: ')
        node2 = tf.add(node2, y_new)
        # node2 = tf.Print(node2, [node2], message='node2: ')

        mul = tf.multiply(node1,node2,name='matmul_op')
        # mul = tf.Print(mul,[mul], message='mul: ', summarize=100)

        # divide = tf.divide(node1, node2, name='divide')
        divide = node1/node2
        # divide = tf.Print(divide, [divide], message='divide: ', summarize=100)

        printout = [mul,divide]
        # printout = tf.Print([mul,divide], [mul,divide],message='mul and divide: ', summarize=100)
        printout = tf.reshape(printout, [-1]) # Flattens array
        # printout = tf.squeeze(printout) # only removes dimension of 1
        # concat = tf.concat(printout, 0, name='concat')
        output = tf.sigmoid(printout)
        print(output.shape)
        # output = tf.Print(output, [output], message='Output: ', summarize=100)
        loss = tf.reduce_mean(tf.square(tf.subtract(tf.cast(tf.constant([1,1,1,1]),tf.float32),output)))
        # loss = tf.losses.mean_squared_error(labels = [1,1,1,1],predictions=printout)
        # reduce_mean calculates average along a give axis; or all elements if no axis is given
        # loss = tf.Print(loss, [loss], message='Loss: ', summarize=100)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        gradients = optimizer.compute_gradients(loss, tf.trainable_variables())
        train = optimizer.apply_gradients(10*gradients)
        # train = optimizer.minimize(loss)
    return x, y, loss, train, output, gradients, g

if __name__ == "__main__":
    x, y, loss, train, output, gradients, g = _build_graph()
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        # feed_dict can be used to overwrite any tensor in the graph;
        # the only difference between placeholders and other tf.Tensors is that placeholders throw an error if no value is fed to them
        for i in range(1000):
            print('iteration', i)
            ret = sess.run([{'loss':loss,'output':output}, train], feed_dict={x:[1, 3], y:[2,4]})[0]
            print('loss', ret['loss'])
            print('output', ret['output'])
        gra = sess.run(gradients,feed_dict={x:[1, 3], y:[2,4]})
        pass
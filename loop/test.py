from __future__ import print_function
#
# import tensorflow as tf
#
# # Basic constant operations
# # The value returned by the constructor represents the output
# # of the Constant op.
# a = tf.constant(2, name='adult')
# b = tf.constant(3)
#
# # Launch the default graph.
# with tf.Session() as sess:
#     print("a=2, b=3")
#     print("Addition with constants: {0}".format(sess.run('adult')))
#     print("Multiplication with constants: %i" % sess.run(a*b))
#
# # Basic Operations with variable as graph input
# # The value returned by the constructor represents the output
# # of the Variable op. (define as input when running session)
# # tf Graph input
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)
#
# # Define some operations
# add = tf.add(a, b)
# mul = tf.mul(a, b)
#
# # Launch the default graph.
# with tf.Session() as sess:
#     # Run every operation with variable input
#     print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
#     print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

import tensorflow as tf

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d, name='example')

with tf.Session() as sess:
    test = sess.run(e)
    print(e.name) #example:0
    test = tf.get_default_graph().get_tensor_by_name("example:0")
    print(test) #Tensor("example:0", shape=(2, 2), dtype=float32)
    blust = sess.run(test)
    print(blust)
from __future__ import division
import tensorflow as tf
import numpy as np


class ConvNN(object):
    def __init__(self, params):
        self.name = params['name']
        self.learn = True
        self.method = params['method']
        self.learning_rate = params['learning_rate']
        self.dtype = tf.float32
        self.stride_size = params['stride_size']
        self.current_frame = None
        self.next_frame = None

        self.patch_size = [params['patch_width'], params['patch_height']]
        self.patch_channels = [params['channels']]  # num of channels - here it's blue green and red i.e. bgr
        self.patch_num_features = [params['num_output']]  # num of output features
        #  - here we want only 1 to get a kernel, if 3 then outputs kernel to every channel
        self.weights_shape = self.patch_size + self.patch_channels + self.patch_num_features
        self.train_op = self.build_graph()

    def build(self):
        pass

    def batch(self):
        pass

    def online_graph_forward(self):

        with tf.name_scope(self.name + '/input'):
            current_frame = tf.placeholder(self.dtype, name='current_frame')

        with tf.name_scope(self.name + '/convolution'):
            with tf.name_scope('weights'):
                weights = self.weight_variable(shape=self.weights_shape)
                self.variable_summaries(weights, name='weights')

            with tf.name_scope('bias'):
                bias = self.bias_variable(shape=self.patch_num_features)
                self.variable_summaries(bias, name='bias')

            with tf.name_scope('predictor'):
                conv = tf.nn.conv2d(current_frame, weights, strides=[1, self.stride_size, self.stride_size, 1], padding='SAME', name='conv')
                preactivate = conv + bias
                activation = tf.minimum(tf.maximum(preactivate, 0), 1, name='predicted_frame')
                self.activation_summary(activation, 'prediction')
            return activation, current_frame

    def online_graph_loss(self, activation):
        with tf.name_scope(self.name + '/input'):
            next_frame = tf.placeholder(self.dtype, name='next_frame')

        with tf.name_scope(self.name + '/loss'):
            epsilon = tf.sub(activation, next_frame, name='epsilon')
            epsilon_squared = tf.square(epsilon, name='epsilon_squared')
            self.activation_summary(epsilon_squared, 'epsilon_squared')
            cost = tf.reduce_mean(epsilon_squared)
            tf.scalar_summary('cost', cost)
        return cost

    def online_graph_train(self, cost):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost, global_step)
        return train_step

    def build_graph(self):
        activation = self.online_graph_forward()
        cost = self.online_graph_loss(activation)
        return self.online_graph_train(cost)

    @staticmethod
    def weight_variable(shape):
        # initial = tf.zeros(shape, dtype=tf.float32)
        initial = tf.random_normal(shape, mean=0.5, stddev=0.1, dtype=tf.float32)
        # initial = tf.zeros(shape, dtype=tf.float32)
        # initial = tf.div(tf.ones(shape, dtype=tf.float32), shape[0]*shape[1])
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    @staticmethod
    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope(name + '_' + 'summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary(name + '_' + 'mean/' + name, mean)
            with tf.name_scope(name + '_' + 'stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(var, mean))))
            tf.scalar_summary(name + '_' + 'stddev/' + name, stddev)
            tf.scalar_summary(name + '_' + 'max/' + name, tf.reduce_max(var))
            tf.scalar_summary(name + '_' + 'min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    @staticmethod
    def activation_summary(var, name):
        tf.histogram_summary(name + '/activations', var)
        tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(var))

    @staticmethod
    def flatten(image):
        return np.ravel(image)


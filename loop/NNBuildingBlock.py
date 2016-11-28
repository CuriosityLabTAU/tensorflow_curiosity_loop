from __future__ import division
import tensorflow as tf


class BuildNN(object):
    def __init__(self, params):
        self.name = params['name']
        self.stride_size = params['stride_size']
        self.scene_batch = []

        self.patch_size = [params['patch_width'], params['patch_height']]
        self.patch_channels = [params['channels']]  # num of channels - here it's blue green and red i.e. bgr
        self.patch_num_features = [params['num_output']]  # num of output features
        #  - here we want only 1 to get a kernel, if 3 then outputs kernel to every channel
        self.num_of_layers = params['num_of_layers']  # TODO: add to params
        self.weights_shape = self.patch_size + self.patch_channels + self.patch_num_features
        self.weights = {}

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

    def variable_summaries(self, var, varname):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary(self.name + '/mean', mean)
            with tf.name_scope(varname + '_' + 'stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(var, mean))))
            tf.scalar_summary(self.name + '/' + 'stddev/' + varname, stddev)
            tf.scalar_summary(self.name + '/' + 'max/' + varname, tf.reduce_max(var))
            tf.scalar_summary(self.name + '/' + 'min/' + varname, tf.reduce_min(var))
            tf.histogram_summary(varname, var)

    @staticmethod
    def activation_summary(var, name):
        tf.histogram_summary(name + '/activations', var)
        tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(var))

    def conv2d(self, x, act, layer_name, num_filters, weights_names):
        # build a 2d convolution layer with activation function 'act' and input x
        with tf.name_scope(layer_name):
            weights = {}
            bias = {}
            predictions = {}
            with tf.name_scope('weights'):
                for i in range(num_filters):
                    name = weights_names[i]
                    with tf.name_scope(name):
                        weights[name] = self.weight_variable(shape=self.weights_shape)
                        # self.variable_summaries(var=weights, varname=self.name + '/weights_' + name)
            with tf.name_scope('bias'):
                for i in range(num_filters):
                    name = weights_names[i]
                    with tf.name_scope(name):
                        bias[name] = self.bias_variable(shape=self.patch_num_features)
                        # self.variable_summaries(var=bias[name], varname='bias')

            with tf.name_scope('predictors'):
                for i in range(num_filters):
                    name = weights_names[i]
                    conv = tf.nn.conv2d(x, weights[name], strides=[1, self.stride_size, self.stride_size, 1], padding='SAME', name=name+'_conv')
                    preactivate = conv + bias[name]
                    predictions[name] = act(preactivate)
                    self.activation_summary(predictions[name], 'prediction_'+name)

        return predictions

    def deconv2d(self, x):
        # a 2d deconvolution layer
        x_shape = tf.shape(x)
        output_shape = tf.pack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        with tf.name_scope('weights'):
            weights = self.weight_variable(shape=self.weights_shape)
            self.variable_summaries(var=weights, varname=self.name + '/weights')
        with tf.name_scope('deconv'):
            deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, self.stride_size, self.stride_size, 1], padding='VALID')

        return deconv

    def fully_connected(self, x, act, actname):
        # build a fully connected layer with activation function 'act' and input x
        with tf.name_scope(self.name + '/full'):
            with tf.name_scope('weights'):
                weights = self.weight_variable(shape=self.weights_shape)  # TODO: check if weights_shape works here
                self.variable_summaries(var=weights, varname=self.name + '/weights')

            with tf.name_scope('bias'):
                bias = self.bias_variable(shape=self.patch_num_features)
                self.variable_summaries(var=bias, varname='bias')

            with tf.name_scope('predictor'):
                full = tf.matmul(x, weights, name='fully connected')
                preactivate = full + bias
                activation = act(preactivate, name=actname)
                self.activation_summary(activation, 'prediction')

        return activation

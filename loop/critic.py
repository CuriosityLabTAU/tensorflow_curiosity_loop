from __future__ import division
import numpy as np
import tensorflow as tf
import NNBuildingBlock
sess = tf.Session


class Critic(NNBuildingBlock.BuildNN):
    def __init__(self, critic_params, graph):

        super(Critic, self).__init__(critic_params)
        self.learn = True
        self.method = critic_params['method']
        self.learning_rate = critic_params['learning_rate']
        self.dtype = tf.float32
        self.stride_size = critic_params['stride_size']
        self.current_frame = None
        self.next_frame = None
        self.validation_threshold = critic_params['validation_threshold']

        self.patch_size = [critic_params['patch_width'], critic_params['patch_height']]
        self.patch_channels = [critic_params['channels']]  # num of channels - here it's blue green and red ie bgr
        self.patch_num_features = [critic_params['num_output']]  # num of output features - here we want only 1 to get a kernel, if 3 then outputs kernel to every channel
        self.weights_shape = self.patch_size + self.patch_channels + self.patch_num_features

        # building the graph TODO: add to graph and add to collections!
        self.image_shape = critic_params['image shape']  # TODO: add to params
        with graph.as_default():
            self.prediction = self.inference()
            self.cost, self.state_value = self.loss(self.prediction)
            self.train_op = self.train(self.cost)
            # self.summary_op = tf.merge_summary([])
        self.perform_learning = self.learning_method()

    def inference(self):
        # build the network structure
        weights_names = ['first', 'second', 'third']
        num_filters = len(weights_names)
        with tf.name_scope(self.name + '/input'):
            self.current_frame = tf.placeholder(self.dtype, shape=self.image_shape, name='current_frame')

        def activator(x):
            return tf.maximum(0, tf.nn.relu(x))

        with tf.name_scope(self.name + '/convolution'):
            # act = activator
            predictions = self.conv2d(self.current_frame, tf.nn.relu, 'conv{0}'.format(0), num_filters, weights_names)

        # with tf.name_scope(self.name + '/convolution{0}'.format(1)):
        #     # act = activator
        #     weights_names = ['1', '2', '3']
        #     prediction = self.conv2d(prediction_0, tf.nn.relu, 'conv{0}'.format(1), num_filters, weights_names)

        prediction = tf.pack([predictions['first'], predictions['second'], predictions['third']])

        return prediction

    def loss(self, prediction):
        # create the loss part

        with tf.name_scope(self.name + '/true_next'):
            self.next_frame = tf.placeholder(self.dtype, shape=self.image_shape, name='next_frame')

        with tf.name_scope(self.name + '/loss'):
            epsilon = tf.sub(prediction, self.next_frame, name='epsilon')
            epsilon_squared = tf.square(epsilon, name='epsilon_squared')
            self.activation_summary(epsilon_squared, 'epsilon_squared')
            cost = tf.reduce_mean(epsilon_squared)
            tf.scalar_summary('cost', cost)
        return cost, epsilon_squared

    def train(self, cost):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            train_step = optimizer.minimize(cost, global_step)
        return train_step

    def state_values(self, state):

        return self.prediction.eval(feed_dict={self.current_frame: state, self.next_frame: None}),\
               self.cost.eval(feed_dict={self.current_frame:state, self.next_frame:None})  # TODO: trouble

    def update_critic(self, state, true_next_state):
        return self.train_op.eval(feed_dict={self.current_frame: state, self.next_frame: None})  # TODO: trouble

    def perform_learning_batch_scene(self, input_states, true_next_states, session, summery_op):
        feed_dict = {self.current_frame: input_states, self.next_frame: true_next_states}
        _, cost, epsilon_squared, prediction, critic_summary = session.run([self.train_op, self.cost, self.state_value, self.prediction, summery_op], feed_dict=feed_dict)
        return prediction, cost, epsilon_squared, critic_summary

    def perform_learning_online(self, input_states, true_next_states, session, summery_op):
        total_scene_cost, error_squared_, state_values_, critic_summaries = 0, [], [], []
        print 'critic cost:',
        for i in range(len(input_states)):
            state, next_state = [input_states[i]], [true_next_states[i]]
            feed_dict = {self.current_frame: np.asarray(state), self.next_frame: np.asarray(next_state)}
            # _, cost, error_squared, prediction, critic_summary = session.run([self.train_op, self.cost, self.state_value, self.prediction, summery_op], feed_dict=feed_dict)
            _, cost, error_squared, prediction = session.run([self.train_op, self.cost, self.state_value, self.prediction], feed_dict=feed_dict)
            print cost,
            error_squared_.append(error_squared)
            state_values_.append(prediction)
            # critic_summaries.append(critic_summary)
            total_scene_cost += cost
        print ''
        # return state_values_, total_scene_cost, error_squared_, critic_summaries
        return state_values_, total_scene_cost, error_squared_

    def learning_method(self):
        if self.method == 'online':
            return self.perform_learning_online
        elif self.method == 'batch_scene':
            return self.perform_learning_batch_scene

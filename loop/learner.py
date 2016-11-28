from __future__ import division
import numpy as np
import CNN
import NNBuildingBlock
import tensorflow as tf

# learner_things = {
#     'learning_rate'
#     'validation_threshold'
#     'method'
#     'wanted_width'
#     'wanted_height'
#     'channels'
#     'patch_width'
#     'patch_height'
#     'num_output'
#     'stride_size'
# }


class Learner(NNBuildingBlock.BuildNN):

    def __init__(self, learner_params, graph):

        super(Learner, self).__init__(learner_params)
        self.learn = True
        self.method = learner_params['method']
        self.learning_rate = learner_params['learning_rate']
        self.dtype = tf.float32
        self.stride_size = learner_params['stride_size']
        self.current_frame = None
        self.next_frame = None

        self.patch_size = [learner_params['patch_width'], learner_params['patch_height']]
        self.patch_channels = [learner_params['channels']]  # num of channels - here it's blue green and red ie bgr
        self.patch_num_features = [learner_params['num_output']]  # num of output features - here we want only 1 to get a kernel, if 3 then outputs kernel to every channel
        self.weights_shape = self.patch_size + self.patch_channels + self.patch_num_features

        # building the graph TODO: add to graph and add to collections!
        self.image_shape = learner_params['image shape']  # TODO: add to params
        with graph.as_default():
            self.prediction = self.inference()
            self.cost, self.epsilon_squared = self.loss(self.prediction)
            self.train_op = self.train(self.cost)
            # self.summary_op = tf.merge_all_summaries()
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

        prediction = tf.squeeze(tf.pack([predictions['first'], predictions['second'], predictions['third']], axis=-1))

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
            train_op = optimizer.minimize(cost, global_step)
        return train_op

    @staticmethod
    def flatten(image):
        return np.ravel(image)

    # def process_input(self, states, session):
    #     feed_dict = {self.current_frame:states, self.next_frame:None}
    #     epsilon, prediction = session.run([self.epsilon_squared, self.prediction], feed_dict=feed_dict)
    #     return epsilon, prediction
    #
    # def update_learner(self, input_states, true_next_states, session, summery_op):
    #     # backprop
    #     feed_dict = {self.current_frame: input_states, self.next_frame: true_next_states}
    #     cost = session.run([self.train_op, self.cost], feed_dict=feed_dict)
    #     return cost

    def perform_learning_batch_scene(self, input_states, true_next_states, session, summery_op):
        feed_dict = {self.current_frame: input_states, self.next_frame: true_next_states}
        _, cost, epsilon_squared, prediction, learner_summary = session.run([self.train_op, self.cost, self.epsilon_squared, self.prediction, summery_op], feed_dict=feed_dict)
        return prediction, cost, epsilon_squared, learner_summary

    def perform_learning_online(self, input_states, true_next_states, session, summery_op):
        total_scene_cost, epsilon_squared_, prediction_, learner_summaries = 0, [], [], []
        print 'learner cost:',
        for i in range(len(input_states)):
            state, next_state = [input_states[i]], [true_next_states[i]]
            feed_dict = {self.current_frame: state, self.next_frame: next_state}
            _, cost, epsilon_squared, prediction, learner_summary = session.run([self.train_op, self.cost, self.epsilon_squared, self.prediction, summery_op], feed_dict=feed_dict)
            print cost,
            epsilon_squared_.append(np.squeeze(epsilon_squared))
            prediction_.append(prediction)
            learner_summaries.append(learner_summary)
            total_scene_cost += cost
        return prediction_, total_scene_cost, epsilon_squared_, learner_summaries

    def learning_method(self):
        if self.method == 'online':
            return self.perform_learning_online
        elif self.method == 'batch_scene':
            return self.perform_learning_batch_scene



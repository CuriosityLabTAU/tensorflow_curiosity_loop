from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from matplotlib import pyplot as plt, style, rcParams
import time
import numpy as np
import world
import agent


style.use('ggplot')
rcParams.update({'font.size': 10})


class Exp(object):
    def __init__(self, num_epochs, movies_dir, num_of_episodes, scene_lists_dir, save_dir, graph_name,
                 world_params, agent_params, name="Experiment", gpu=False):
        self.name = name
        self.continue_learning = True  # while True the experiment will keep running
        self.learner_proceed = True  # while True the learner will keep learning from new data
        self.critic_proceed = False  # while True the critic will keep learning from new data
        self.num_epochs = num_epochs
        self.load_dir = movies_dir
        self.num_episodes = num_of_episodes
        self.current_episode = 0
        self.episode_list = [movies_dir + 'S06E{0}.mkv'.format(i + 1) for i in range(num_of_episodes)]
        self.scene_lists_dir = scene_lists_dir
        self.scene_lists = self.load_scene_lists(num_of_episodes)  # create the scene lists needed
        self.save_dir = save_dir
        self.graph_name = graph_name

        self.start_time = None
        self.step_count = {'episodes_count': 0, 'scenes_count': 0, 'frames_count': 0, 'duration': 0}
        # self.figure = self.create_figure()
        self.time_step = 0

        world_params['episode_list'] = self.episode_list
        world_params['scene_lists'] = self.scene_lists
        world_params['num_episodes'] = self.num_episodes

        self.summary_op, self.summary_writer = None, None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.world, self.agent = self.create_players(world_params, agent_params)
            self.saver = tf.train.Saver()

    def run(self):

        with tf.Session(graph=self.graph) as sess:  # runs the session with self.graph as the graph
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(self.save_dir, sess.graph)
            init = tf.initialize_all_variables()
            sess.run(init)

            while self.continue_learning:
                # TODO: decide how to perform online or collect the scene first. what is the order we want
                self.world.update_state()  # changes the current state of the world (world.current_state)
                if self.world.ret:  # check that there is another scene
                    self.time_step += 1
                    # collect = not self.world.new_scene  # if its a new scene than we stop collecting and start learning
                    self.world.display_state()  # display the current state of the world
                    if not self.world.new_scene:
                        self.agent.collect_scene(self.world.current_state)
                    else:
                        # summary = self.agent.run_loop(self.world.current_state, sess)  # input: current world state, output: action the agent wants to do
                        summaries = self.agent.run_loop(sess, self.summary_op)  # input: current world state, output: action the agent wants to do
                        # self.summary_writer.flush()
                        if agent.learner.learn:
                            self.summary_writer.add_summary(summaries['learner'], self.time_step)
                        if agent.critic.learn:
                            self.summary_writer.add_summary(summaries['critic'], self.time_step)
                    if self.time_step % 1000 == 999:
                        self.saver.save(sess, self.save_dir + self.graph_name + '_checkpoint{0}'.format(self.time_step))
                else:
                    break

                self.summary_writer.close()

    def init_exp(self):
        self.start_time = time.time()

    def create_players(self, world_things, agent_things):
        real_world = world.World(world_things)
        real_agent = agent.Agent(agent_things, self.graph)
        return real_world, real_agent

    # def create_figure(self):
    #     fig = plt.figure(figsize=(8, 7))
    #     fig.suptitle(
    #         r'$\alpha$' + ' = {:.2e}, scene count = {1}, duration = {2}'.format(
    #             self.learner.learning_rate, self.step_count['scenes_count'], self.step_count['duration']))
    #     return fig

    def load_scene_lists(self, num_episodes):
        scene_lists = []
        for episode in range(num_episodes):
            filename = 'S06E{0}_scenes.npz'.format(episode + 1)
            new = np.load(self.scene_lists_dir + filename)
            scene_lists.append(new)
        return scene_lists


    # def draw_fig(self):
    #     self.figure.clear()
    #     plt.pause(0.00001)
    #     plt.show()


import learner, critic, actor
import numpy as np


class Agent(object):
    def __init__(self, agent_params, graph):
        actor_params = agent_params['actor_params']
        learner_params = agent_params['learner_params']
        critic_params = agent_params['critic_params']
        self.learner = learner.Learner(learner_params, graph)  # ConvNN
        self.critic = critic.Critic(critic_params, graph)  # ConvNN
        self.actor = actor.Actor(actor_params, graph)  # RL
        self.learner.learn = agent_params['who_learns']['learner']  # if True, learner updates weights
        self.critic.learn = agent_params['who_learns']['critic']  # if True, critic updates weights
        self.scene_states = []

        self.prediction = None
        self.epsilon_squared = []
        self.critic_value = None
        self.action_map = None

    def run_loop(self, session, summery_op):
        # perform a step of the agent. input the state of the world and output prediction, value and an actions map.
        # prediction is the image the learner predicts will be next.
        # value is the loss function value of the critic TODO: (or the prediction error of the entire image),
        # and actions map is the map of actions the policy recommends in each pixel.
        # input_states, true_next_states = self.learner.process_input(self.scene_states, session)  # batch
        summaries = {}
        input_states, true_next_states = self.process_input()  # batch into the corresponding inputs
        print('scene length is {0}'.format(len(input_states)))
        if self.learner.learn:
            self.prediction, total_learner_value, self.epsilon_squared, summaries['learner']\
                = self.learner.perform_learning(input_states, true_next_states, session, summery_op)  # epsilon is current reward
            average_learner_cost_in_scene = total_learner_value / len(input_states)
            print('the average learner cost in scene is {0}'.format(average_learner_cost_in_scene))

        if self.critic.learn:
            # self.critic_value = self.critic.update_critic(self.scene_states, self.epsilon_squared)
            # self.critic_value, summaries['critic'] = self.critic.perform_learning(input_states, self.epsilon_squared, session, summery_op)
            self.critic_value = self.critic.perform_learning(input_states, self.epsilon_squared, session)
        self.action_map = self.actor.select_action(self.critic_value)  # TODO: complete actor code
        if self.actor.learn:
            self.actor.update_policy()

        return summaries

    def collect_scene(self, frame):
        # Gathers the scene into a 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        # temp = frame[np.newaxis, :, :, :]  # TODO: check necessity
        # self.scene_states.append(temp)
        self.scene_states.append(frame / 255)

    def process_input(self):
        input_states = self.scene_states[0:-1]
        true_next_states = self.scene_states[1:]
        self.scene_states = []  # clean scene states
        return input_states, true_next_states

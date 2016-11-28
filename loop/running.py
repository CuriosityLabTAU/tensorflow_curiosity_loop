import experiment

# This is the main file for running an experiment

num_epochs = 1
current_epoch = 1
movies_dir = '/home/goren/projects/vision/movies/'
num_of_episodes = 1
scene_lists_dir = '/home/goren/projects/vision/scene_generator/'
save_dir = '/home/goren/PycharmProjects/tensorflow_curiosity_loop/data/'
gpu = False
graph_name = 'vision_loop'
methods = ['online', 'scene_batch']

image_width, image_height = 840, 480
wanted_frame_width = 320
wanted_frame_height = int(image_height * wanted_frame_width / image_width)
wanted_fps = 5
with_faces_only = True
stop_learning_margin = 1

world_params = {
    'wanted_frame_width': wanted_frame_width,
    'wanted_fps': wanted_fps,
    'with_faces_only': with_faces_only,
    'real_image_shape': (image_width, image_height)

}


learner_num_output = 1  # if we want a number diff then 1 and three then need to change epsilon calculation
learner_stride_size = 1
learner_channels = 3
learner_learning_rate_not_scaled = 0.01
learner_validation_threshold = 0.01
learner_patch_width = 17  # the width of the sliding window frame
learner_patch_height = 17  # the height of the sliding window frame
learner_method = methods[0]  # possible methods are 'scene_batch' and 'online'
learner_name = 'learner'
num_of_layers = 2


learner_params = {
    'name': learner_name,
    'learning_rate': learner_learning_rate_not_scaled/(wanted_frame_width*wanted_frame_height),
    'validation_threshold': learner_validation_threshold,
    'method': learner_method,
    'wanted_width': wanted_frame_width,
    'wanted_height': wanted_frame_height,
    'channels': learner_channels,
    'patch_width': learner_patch_width,
    'patch_height': learner_patch_height,
    'num_output': learner_num_output,
    'stride_size': learner_stride_size,
    'num_of_layers': num_of_layers,
    'image shape': (None, wanted_frame_height, wanted_frame_width, 3)  # TODO: change to size of downgraded frame? YES!!
}

critic_num_output = 1  # if we want a number diff then 1 and three then need to change epsilon calculation
critic_stride_size = 1
critic_channels = 3
critic_learning_rate_not_scaled = 0.01
critic_validation_threshold = 0.01
critic_patch_width = 9  # the width of the sliding window frame
critic_patch_height = 9  # the height of the sliding window frame
critic_method = 'online'  # others are 'scene_batch', 'fixed_batch'
critic_name = 'critic'

critic_params = {
    'name': critic_name,
    'learning_rate': critic_learning_rate_not_scaled/(wanted_frame_width*wanted_frame_height),
    'validation_threshold': critic_validation_threshold,
    'method': critic_method,
    'wanted_width': wanted_frame_width,
    'wanted_height': wanted_frame_height,
    'channels': critic_channels,
    'patch_width': critic_patch_width,
    'patch_height': critic_patch_height,
    'num_output': critic_num_output,
    'stride_size': critic_stride_size,
    'num_of_layers': 1,
    'image shape': (None, wanted_frame_height, wanted_frame_width, 3)  # downsized!!!
}

max_step_size = k = 5
moves = ['up', 'right', 'down', 'left', 'stay']
step_sizes = range(k+1)[1:-1]

actor_params = {
    'moves': moves,
    'step_sizes': step_sizes
}

who_learns = {'learner': True, 'critic': False}  # starting position of learning: who learns at the start of experiment

agent_params = {
    'learner_params': learner_params,
    'critic_params': critic_params,
    'actor_params': actor_params,
    'who_learns': who_learns,
    'stop_learning_margin': stop_learning_margin
}
exp = experiment.Exp(num_epochs, movies_dir, num_of_episodes, scene_lists_dir, save_dir, graph_name,
                     world_params, agent_params, gpu=gpu)

while current_epoch <= num_epochs:
    exp.run()


# exp.draw_fig()

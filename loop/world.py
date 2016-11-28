import numpy as np
import cv2
import faces as f


class World(object):
    # this is the world agent. it is essentially the video holder with all the necessary attributes
    def __init__(self, world_params, name='World'):
        self.name = name
        self.current_episode = -1

        self.current_scene = None
        self.current_frame = None
        self.cap = None
        self.scene_starts = []
        self.scene_ends = []
        self.last_frame = None
        self.jump = None
        self.fps = None
        self.scene_first_frame = None
        self.scene_last_frame = None

        self.num_episodes = world_params['num_episodes']
        self.episode_list = world_params['episode_list']
        self.lists = world_params['scene_lists']
        self.wanted_frame_width = int(world_params['wanted_frame_width'])
        image_width, image_height = world_params['real_image_shape']
        # self.wanted_frame_height = int(image_width * self.wanted_frame_width / image_height)
        self.wanted_frame_height = int(image_height * self.wanted_frame_width / image_width)  # the correct one
        self.dim = (self.wanted_frame_width, self.wanted_frame_height)
        self.wanted_fps = world_params['wanted_fps']
        self.faces = world_params['with_faces_only']
        self.load_next_episode()
        self.test_set = [[]*self.num_episodes]
        self.validation_set = [[]*self.num_episodes]
        self.current_state = None  # won't be normalized!!! normalization is in the agent/learner
        self.ret = True
        # the next two are needed for the agent
        self.new_episode = True
        self.new_scene = True
        self.scene_ended = True

        self.face_detect = f.FaceDetection('haarcascade_frontalface_default.xml')

    def update_state(self, action=None):
        # make the next frame needed available with the proper parameters TODO: add exit strategy when self.ret = False
        self.set_next_frame()
        self.ret, frame = self.cap.read()
        # while self.face_detect.detect_from_color(frame) is None:
        #     self.set_next_frame()
        #     self.ret, frame = self.cap.read()
        # self.current_state = self.normalize(self.downgrade(frame))
        self.current_state = self.downgrade(frame)

    def downgrade(self, image):
        # resize the extracted image to the correct parameters
        return cv2.resize(image, self.dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize(image):
        # change all entries of the frame to values in [0, 1]
        mini, maxi = min(0, np.min(image)), max(np.amax(image),255)  # TODO: check if necessary
        return (image - mini) / (maxi - mini)

    def set_next_frame(self):
        # find the correct next frame (changes in wanted fps and scene lists make this step necessary).
        # this function is also used in order to generate the validation set.
        attempt = self.current_frame + self.jump

        if attempt >= self.last_frame:
            self.load_next_episode()
            self.new_scene, self.new_episode = True, True
        elif attempt >= self.scene_last_frame:
            self.load_next_scene()
            self.new_scene, self.new_episode = True, False
        else:
            self.current_frame = attempt
            self.new_scene, self.new_episode = False, False

        self.cap.set(1, self.current_frame)

        self.test_set[self.current_episode].append(self.current_frame)
        self.validation_set[self.current_episode].append(self.current_frame + 1)

    def load_next_episode(self):
        # move the world to the next episode in the video. change all attributes relevant for this
        self.current_episode += 1
        if self.current_episode <= self.num_episodes:  # TODO: check for correctness
            self.current_scene = 0
            self.cap = cv2.VideoCapture(self.episode_list[self.current_episode])
            self.scene_starts = self.lists[self.current_episode]['scene_list']
            self.scene_ends = self.scene_starts[1:]
            self.scene_ends = np.append(self.scene_ends, self.cap.get(7))
            if self.faces:
                # TODO: check if "self.lists[self.current_episode]['scene_faces']" is correct
                self.scene_starts = [self.scene_starts[i] for i, e in enumerate(self.lists[self.current_episode]['scene_faces']) if e > 0]
                self.scene_ends = [self.scene_ends[i] for i, e in enumerate(self.lists[self.current_episode]['scene_faces']) if e > 0]
            self.last_frame = self.scene_ends[-1]
            self.jump = int(self.cap.get(5) / self.wanted_fps)
            self.fps = self.cap.get(5) / self.jump
            self.scene_first_frame = self.scene_starts[self.current_scene]
            self.scene_last_frame = self.scene_starts[self.current_scene + 1] - 1
            self.current_frame = self.scene_first_frame

    def only_faces(self, frame):
        return len(self.face_detect.detect_from_color(frame))

    def load_next_scene(self):
        # move the world to the next scene in the video. change all attributes relevant for this
        self.current_scene += 1
        self.scene_first_frame = self.scene_starts[self.current_scene]
        self.scene_last_frame = self.scene_starts[self.current_scene + 1] - 1
        self.current_frame = self.scene_first_frame

    def clean_validation_set(self):
        # when we run the algorithm with fps less than the true fps we ignore a lot of data that comes from
        # exactly the same distribution as the train set. this data is the perfect validation set for our model.
        # This function cleans the validation set generated throghout the process by making sure that we don't use
        # the training set examples. it will run only once in the experiment phase.
        for i in range(self.num_episodes):
            temp = [frame_num for frame_num in self.validation_set[i] if frame_num not in self.test_set[i]]
            self.validation_set[i] = temp
        return self.validation_set

    def display_state(self):
        # cv2.imshow('episode    {0}'.format(self.episode_list[self.current_episode - 1]), cv2.cvtColor(self.current_state, cv2.COLOR_RGB2BGR))
        cv2.imshow('episode  {0}'.format(self.episode_list[self.current_episode - 1]), self.current_state)
        # TODO change self.current_state to frame downgraded
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()





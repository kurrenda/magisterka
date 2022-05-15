from abc import ABC, abstractmethod, abstractproperty

import mediapipe as mp


class Creator(ABC):
    @abstractmethod
    def get_feature(self):
        pass


class Feature():
    @abstractproperty
    def solution(self):
        pass

    @abstractproperty
    def solution_object(self):
        pass

    @abstractproperty
    def drawing(self):
        pass

    @abstractmethod
    def draw(self):
        pass


class MediaPipeCreator(Creator):
    def __init__(self, name):
        self.name = name

    def create_feature(self, name):
        if name == 'hands':
            return MediaPipeHands()


class MediaPipeHands(Feature):
    solution = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    @property
    def solution_object(self):
        return self.solution.Hands(static_image_mode=False,
                                   max_num_hands=2,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

    def draw(self, img, hand_lms):
        self.drawing.draw_landmarks(
            img, hand_lms, self.solution.HAND_CONNECTIONS)

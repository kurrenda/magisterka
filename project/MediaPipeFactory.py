from abc import ABC, abstractmethod, abstractproperty
import math
import mediapipe as mp


class Creator(ABC):
    @abstractmethod
    def create_feature(self):
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
    def __init__(self):
        pass

    def create_feature(self, name):
        if name == 'hands':
            return MediaPipeHands()


class MediaPipeHands(Feature):
    MIDDLE_FINGER_MCP = 9
    THUMB_CMC = 1
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    INDEX_FINGER_MCP = 5
    RING_FINGER_MCP = 13
    PINKY_MCP = 17

    solution = mp.solutions.hands
    hands = solution.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils
    saved_hand_landmarks = []
    draw_landmarks = True

    def draw(self, img, hand_lms):
        if self.draw_landmarks:
            self.drawing.draw_landmarks(
                img, hand_lms, self.solution.HAND_CONNECTIONS)

    @staticmethod
    def calculate_distance(x, y, x2, y2):
        distance = math.hypot(x2 - x, y2 - y)
        return distance

    @staticmethod
    def normalize_coordinates(coords):
        key_points_available = [
            MediaPipeHands.MIDDLE_FINGER_MCP,
            MediaPipeHands.WRIST,
            MediaPipeHands.THUMB_TIP,
            MediaPipeHands.INDEX_FINGER_TIP,
            MediaPipeHands.MIDDLE_FINGER_TIP,
            MediaPipeHands.RING_FINGER_TIP,
            MediaPipeHands.PINKY_TIP,
            MediaPipeHands.INDEX_FINGER_MCP,
            MediaPipeHands.RING_FINGER_MCP,
            MediaPipeHands.PINKY_MCP,
        ]
        hand_size = MediaPipeHands.calculate_distance(
            coords[MediaPipeHands.WRIST][0],
            coords[MediaPipeHands.WRIST][1],
            coords[MediaPipeHands.MIDDLE_FINGER_MCP][0],
            coords[MediaPipeHands.MIDDLE_FINGER_MCP][1]
        )

        distances = [
            (4, 8),
            (8, 12),
            (12, 16),
            (16, 20),
            (0, 4),
            (0, 8),
            (0, 16),
            (0, 20)
        ]
        total = 0
        for i, j in distances:
            total += MediaPipeHands.calculate_distance(
                coords[i][0], coords[i][1], coords[j][0], coords[j][1])/hand_size

        return total

    @staticmethod
    def get_error(result, gesture_result):
        return abs(gesture_result-result)

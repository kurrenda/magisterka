from abc import ABC, abstractmethod, abstractproperty
from MediaPipeFileManager import MediaPipeFileManager
import time
import glob as glob
import math
import mediapipe as mp
import cv2
import os
import numpy as np


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
        elif name == 'background':
            return MediaPipeBackground()


class MediaPipeBackground(Feature):
    BG_COLOR = (192, 192, 192)
    solution = mp.solutions.selfie_segmentation
    solution_object = solution.SelfieSegmentation()
    drawing = mp.solutions.drawing_utils

    def change_background(self, image_rgb):
        results = self.solution_object.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        bg_image = cv2.GaussianBlur(image, (55, 55), 0)

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = self.BG_COLOR

        output_image = np.where(condition, image, bg_image)
        return output_image


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
    solution_object = solution.Hands(static_image_mode=False,
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

    def import_gestures_distances(self, gestures_path_pattern, mp_hands, camera):
        imported_gestures = {}
        for file in glob.glob(gestures_path_pattern):
            trained_gesture_landmarks = MediaPipeFileManager.load_from_image(
                file,
                raw=False,
                height=camera.height,
                width=camera.width
            )
            imported_gestures[os.path.basename(file)] = mp_hands.normalize_coordinates(
                trained_gesture_landmarks)
        return imported_gestures

    def find_gesture(self, own_gesture_distance, gestures_distances, mp_hands, gesture_timestamp):
        errors = {}
        for key, distance in gestures_distances.items():
            error = mp_hands.get_error(own_gesture_distance, distance)
            errors[key] = error
        min_error_name = min(errors, key=errors.get)

        tolerance = 0.6
        if errors[min_error_name] < tolerance:
            name = min_error_name
        else:
            name = "Not recognized"

        hold_seconds = 1
        if not gesture_timestamp:
            gesture_timestamp.append((name, time.time()))
        else:
            if name != gesture_timestamp[0][0]:
                gesture_timestamp.pop(0)
                gesture_timestamp.append((name, time.time()))
            # seconds to hold gesture to get recognized
            if time.time() - gesture_timestamp[0][1] >= hold_seconds:
                return name
        return None

    def train_gesture(self):
        manager = MediaPipeFileManager('train')
        manager.export_images_to_csv(r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\training_data\*.png')
        print('Successfully exported data')

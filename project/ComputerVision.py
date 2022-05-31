import cv2
import mediapipe as mp
import time
import glob
import os
from ohbot import ohbot
from Camera import Camera
from MediaPipeFactory import MediaPipeCreator
from MediaPipeFileManager import MediaPipeFileManager


class ComputerVision():
    mediapipe_factory = MediaPipeCreator()
    camera = Camera()

    def __init__(self,  is_tracking=False):
        self.is_tracking = is_tracking

    def import_gestures_distances(self, gestures_path_pattern, mp_hands):
        imported_gestures = {}
        for file in glob.glob(gestures_path_pattern):
            trained_gesture_landmarks = MediaPipeFileManager.load_from_image(
                file,
                raw=False,
                height=self.camera.height,
                width=self.camera.width
            )
            imported_gestures[os.path.basename(file)] = mp_hands.normalize_coordinates(
                trained_gesture_landmarks)
        return imported_gestures

    def find_gesture(self, own_gesture_distance, gestures_distances, mp_hands):
        errors = {}
        for key, distance in gestures_distances.items():
            error = mp_hands.get_error(own_gesture_distance, distance)
            errors[key] = error
        min_error_name = min(errors, key=errors.get)
        
        tolerance = 0.6
        if errors[min_error_name] < tolerance:
            return min_error_name
        else:
            return "Not recognized"

    def start(self):
        mp_hands = self.mediapipe_factory.create_feature('hands')

        if self.is_tracking:
            self.camera.set_start_position()

        gestures_distances = self.import_gestures_distances(
            r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\photos\*.png', mp_hands)

        print(gestures_distances)

        while True:
            self.camera.read_image()
            results = mp_hands.hands.process(self.camera.imgRGB)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_hands.saved_hand_landmarks = []
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(
                            lm.x * self.camera.width), int(lm.y*self.camera.height)
                        mp_hands.saved_hand_landmarks.append((cx, cy))
                        if self.is_tracking:
                            self.camera.fix_position(id, cx, cy)

                    mp_hands.draw(self.camera.img, hand_lms)
                    current_gesture = mp_hands.normalize_coordinates(
                        mp_hands.saved_hand_landmarks)

                    gesutre = self.find_gesture(current_gesture, gestures_distances, mp_hands)
                    self.camera.put_text(gesutre)

                    key = cv2.waitKey(1)
                    if key % 256 == 97:
                        self.camera.export_image('kciuk')
                    elif key % 256 == 98:
                        self.camera.export_image('otwarta_reka')
                    elif key % 256 == 99:
                        self.camera.export_image('piesc')
                    elif key % 256 == 32:
                        mp_hands.draw_landmarks = not mp_hands.draw_landmarks

            self.camera.show()

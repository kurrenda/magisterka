import cv2
import mediapipe as mp
import time
import glob
from datetime import datetime
import os
import numpy as np
import joblib
from ohbot import ohbot
from Camera import Camera
from MediaPipeFactory import MediaPipeCreator
from MediaPipeFileManager import MediaPipeFileManager


class ComputerVision():
    mediapipe_factory = MediaPipeCreator()
    camera = Camera()

    def __init__(self,  is_tracking=False):
        self.is_tracking = is_tracking
        self.is_custom_model = False
        self.custom_model = joblib.load(
            fr'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\models\one_class_model.pkl')

    def start(self):
        mp_hands = self.mediapipe_factory.create_feature('hands')
        mp_background = self.mediapipe_factory.create_feature('background')

        if self.is_tracking:
            self.camera.set_start_position()

        gestures_distances = mp_hands.import_gestures_distances(
            r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\photos\*.png',
            mp_hands,
            self.camera
        )

        gesture_timestamp = []

        while True:
            self.camera.read_image()
            hands_results = mp_hands.solution_object.process(
                self.camera.imgRGB)
            self.camera.img = mp_background.change_background(
                self.camera.imgRGB)

            if hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    mp_hands.saved_hand_landmarks = []
                    mp_landmarks = []
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(
                            lm.x * self.camera.width), int(lm.y*self.camera.height)
                        mp_landmarks.append(lm.x)
                        mp_landmarks.append(lm.y)
                        mp_hands.saved_hand_landmarks.append((cx, cy))
                        if self.is_tracking:
                            self.camera.fix_position(id, cx, cy)

                    mp_hands.draw(self.camera.img, hand_lms)
                    current_gesture = mp_hands.normalize_coordinates(
                        mp_hands.saved_hand_landmarks)

                    gesutre = mp_hands.find_gesture(
                        current_gesture, gestures_distances, mp_hands, gesture_timestamp
                    )
                    if gesutre and mp_hands.draw_landmarks:
                        if self.is_custom_model:
                            test_landmarks = np.array(
                                mp_landmarks
                            ).reshape(1, -1)
                            prediction = self.custom_model.predict(
                                test_landmarks)
                            training_dir = r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\training_data'
                            classes = [ item for item in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, item)) ]   
                            
                            if prediction == 0:
                                self.camera.put_text(classes[0])
                            elif prediction == 1:
                                self.camera.put_text(classes[1])
                        else:
                            self.camera.put_text(gesutre)
            else:
                if gesture_timestamp:
                    gesture_timestamp.pop(0)

            key = cv2.waitKey(1)
            if key % 256 == 97:
                self.camera.export_image('kciuk', 'photos')
            elif key % 256 == 98:
                self.camera.export_image('otwarta_reka', 'photos')
            elif key % 256 == 99:
                self.camera.export_image('piesc', 'photos')
            elif key % 256 == 122:
                f_name = f'gesture_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}'
                self.camera.export_image(f_name, 'training_data', 'open')
            elif key % 256 == 116:
                f_name = f'gesture_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}'
                self.camera.export_image(f_name, 'training_data', 'close')
            elif key % 256 == 32:
                mp_hands.draw_landmarks = not mp_hands.draw_landmarks
            elif key % 256 == 120:
                self.is_custom_model = not self.is_custom_model

            self.camera.show()

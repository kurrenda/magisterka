import cv2
import mediapipe as mp
import time
import glob
from datetime import datetime
import os
import numpy as np
import joblib
from Camera import Camera
from MediaPipeFactory import MediaPipeCreator
from MediaPipeFileManager import MediaPipeFileManager
import config


class ComputerVision():
    mediapipe_factory = MediaPipeCreator()
    camera = Camera()

    def __init__(self,  is_tracking=False):
        self.is_tracking = is_tracking
        self.is_custom_model = False
        self.is_background = False
        self.is_cartoon = False
        self.last_gesture = None
        self.custom_model = joblib.load(
            f'{config.MODELS_FOLDER_PATH}/{config.MODEL_FILENAME}.pkl')

        self.mp_hands = self.mediapipe_factory.create_feature('hands')
        self.mp_background = self.mediapipe_factory.create_feature(
            'background')

        if self.is_tracking:
            self.camera.set_start_position()

        self.gestures_distances = None
        self.load_gestures_distances()

        self.gesture_timestamp = []

    def load_gestures_distances(self):
        self.gestures_distances = self.mp_hands.import_gestures_distances(

            f'{config.CLASSES_PHOTOS_PATH}/*.png',
            self.mp_hands,
            self.camera
        )

    def start(self):
        while True:
            self.camera.read_image()
            hands_results = self.mp_hands.solution_object.process(
                self.camera.imgRGB)
            self.camera.img = self.mp_background.change_background(
                self.camera.imgRGB)

            if hands_results.multi_hand_landmarks:
                for hand_lms in hands_results.multi_hand_landmarks:
                    self.mp_hands.saved_hand_landmarks = []
                    mp_landmarks = []
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(
                            lm.x * self.camera.width), int(lm.y*self.camera.height)
                        mp_landmarks.append(lm.x)
                        mp_landmarks.append(lm.y)
                        self.mp_hands.saved_hand_landmarks.append((cx, cy))
                        if self.is_tracking:
                            self.camera.fix_position(id, cx, cy)

                    self.mp_hands.draw(self.camera.img, hand_lms)
                    current_gesture = self.mp_hands.normalize_coordinates(
                        self.mp_hands.saved_hand_landmarks)

                    gesutre = self.mp_hands.find_gesture(
                        current_gesture, self.gestures_distances, self.mp_hands, self.gesture_timestamp
                    )
                    if gesutre and self.mp_hands.draw_landmarks:
                        if self.is_custom_model:
                            test_landmarks = np.array(
                                mp_landmarks
                            ).reshape(1, -1)
                            prediction = self.custom_model.predict(
                                test_landmarks)
                            classes = [item for item in os.listdir(config.TRAINING_DATA_PATH) if os.path.isdir(
                                os.path.join(config.TRAINING_DATA_PATH, item))]

                            for c in range(len(classes)):
                                if prediction == c:
                                    self.camera.put_text(classes[c])
                        else:
                            self.camera.put_text(gesutre)
            else:
                if self.gesture_timestamp:
                    self.gesture_timestamp.pop(0)

            key = cv2.waitKey(1)
            if key % 256 == 97:
                self.camera.export_image('kciuk', config.CLASSES_PHOTOS_PATH)
            elif key % 256 == 98:
                self.camera.export_image(
                    'otwarta_reka',  config.CLASSES_PHOTOS_PATH)
            elif key % 256 == 99:
                self.camera.export_image('piesc',  config.CLASSES_PHOTOS_PATH)
            elif key % 256 == 122:
                f_name = f'gesture_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}'
                self.camera.export_image(
                    f_name,  config.TRAINING_DATA_PATH, 'open')
            elif key % 256 == 116:
                f_name = f'gesture_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}'
                self.camera.export_image(
                    f_name, config.TRAINING_DATA_PATH, 'close')
            elif key % 256 == 32:
                self.mp_hands.draw_landmarks = not self.mp_hands.draw_landmarks
            elif key % 256 == 120:
                self.is_custom_model = not self.is_custom_model
            self.camera.show()

    def capture_frame(self):
        self.camera.read_image()
        hands_results = self.mp_hands.solution_object.process(
            self.camera.imgRGB)

        if self.is_background:
            self.camera.img = self.mp_background.change_background(
                self.camera.imgRGB)
        if self.is_cartoon:
            self.camera.img = self.mp_background.cartoon_background(
                self.camera.img
            )

        if hands_results.multi_hand_landmarks:
            for hand_lms in hands_results.multi_hand_landmarks:
                self.mp_hands.saved_hand_landmarks = []
                mp_landmarks = []
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(
                        lm.x * self.camera.width), int(lm.y*self.camera.height)
                    mp_landmarks.append(lm.x)
                    mp_landmarks.append(lm.y)
                    self.mp_hands.saved_hand_landmarks.append((cx, cy))
                    if self.is_tracking:
                        self.camera.fix_position(id, cx, cy)

                self.mp_hands.draw(self.camera.img, hand_lms)
                current_gesture = self.mp_hands.normalize_coordinates(
                    self.mp_hands.saved_hand_landmarks)

                gesture = self.mp_hands.find_gesture(
                    current_gesture, self.gestures_distances, self.mp_hands, self.gesture_timestamp
                )
                if gesture and self.mp_hands.draw_landmarks:
                    if self.is_custom_model:
                        test_landmarks = np.array(
                            mp_landmarks
                        ).reshape(1, -1)
                        prediction = self.custom_model.predict(
                            test_landmarks)
                        classes = [item for item in os.listdir(config.TRAINING_DATA_PATH) if os.path.isdir(
                            os.path.join(config.TRAINING_DATA_PATH, item))]

                        for c in range(len(classes)):
                            if prediction == c:
                                self.camera.put_text(classes[c])
                    else:
                        if gesture == 'otwarta_reka' and self.last_gesture != gesture:
                            self.is_background = True
                        elif gesture == 'piesc' and self.last_gesture != gesture:
                            self.is_background = False
                            self.is_cartoon = False
                        elif gesture == 'kciuk' and self.last_gesture != gesture:
                            self.is_cartoon = True
                        self.camera.put_text(gesture)
                    self.last_gesture = gesture
        else:
            if self.gesture_timestamp:
                self.gesture_timestamp.pop(0)

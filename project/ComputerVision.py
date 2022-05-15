import cv2
import mediapipe as mp
import time
from ohbot import ohbot
from Camera import Camera
from MediaPipeFactory import MediaPipeCreator

class ComputerVision():
    mediapipe_factory = MediaPipeCreator()
    camera = Camera()

    def start(self):
        mp_hands = self.mediapipe_factory.create_feature('hands')
        ohbot.move(0, self.camera.position)

        while True:
            results = mp_hands.solution.process(self.camera.imgRGB)
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(
                            lm.x * self.camera.width), int(lm.y*self.camera.height)
                        self.camera.draw_circle(cx, cy)

                        if id == 9:
                            if cx < self.camera.width_center - 30:
                                if self.camera.position > 0:
                                    self.camera.position -= 0.15
                            elif cx > self.camera.width_center + 30:
                                if self.camera.position < 10:
                                    self.camera.position += 0.15
                        ohbot.move(0, self.camera.position)

                    mp_hands.draw(self.camera.img, hand_lms)

                    

import cv2
import mediapipe as mp
import time
import csv
import joblib
from datetime import datetime
from argparse import ArgumentParser

from utils.MediaPipeFileManager import MediaPipeFileManager


def capture_gesture():
    cam = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    while True:
        success, img = cam.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        key = cv2.waitKey(1)

        if key%256 == 32:
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    landmarks = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x *w), int(lm.y*h)
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    img_name = f"gesture_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.png"

                    cv2.imwrite(f'./data/photos/{img_name}', img)
                    print(f"{img_name} written!")

        cv2.imshow("Image", img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--save", action="store_true")

    args = parser.parse_args()

    if args.save:
        gesture_file = MediaPipeFileManager('coordinates')
        gesture_file.load_from_images(r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\photos\*.png')
    else:
        capture_gesture()



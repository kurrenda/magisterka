import csv
import glob
import cv2
import mediapipe as mp
from datetime import datetime


class GestureFile:
    # https://google.github.io/mediapipe/solutions/hands
    HAND_LANDMARKS = {
        'wrist': 0,
        'thumb_cmc': 1,
        'thumb_mcp': 2,
        'thumb_ip': 3,
        'thumb_tip': 4,
        'index_finger_mcp': 5,
        'index_finger_pip': 6,
        'index_finger_dip': 7,
        'index_finger_tip': 8,
        'middle_finger_mcp': 9,
        'middle_finger_pip': 10,
        'middle_finger_dip': 11,
        'middle_finger_tip': 12,
        'ring_finger_mcp': 13,
        'ring_finger_pip': 14,
        'ring_finger_dip': 15,
        'ring_finger_tip': 16,
        'pinky_mcp': 17,
        'pinky_pip': 18,
        'pinky_dip': 19,
        'pinky_tip': 20
    }

    def __init__(self, name):
        self.path = f'./data/positions/{name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}.csv'
        self._create_csv_file()

    def _create_csv_file(self):
        header = self._prepare_header(self.HAND_LANDMARKS)
        with open(self.path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)

    def _prepare_header(self, header_data):
        f1 = lambda x: f'{x}_x'
        f2 = lambda x: f'{x}_y'
        return [f(x) for x in header_data.keys() for f in (f1,f2)]

    def load_from_images(self, directory_path_pattern):
        for file in glob.glob(directory_path_pattern):
            self.append_landmarks_positions(file)

    def append_landmarks_positions(self, file_path):
        img = cv2.flip(cv2.imread(file_path), 1)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                if landmarks:
                    self.save_to_file(landmarks)

    def save_to_file(self, data):
        with open(self.path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(data)

    
import csv
import glob
import cv2
import mediapipe as mp
from datetime import datetime


class MediaPipeFileManager:
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
        def f1(x): return f'{x}_x'
        def f2(x): return f'{x}_y'
        return [f(x) for x in header_data.keys() for f in (f1, f2)]

    def export_images_to_csv(self, directory_path_pattern):
        for file in glob.glob(directory_path_pattern):
            raw_landmarks = MediaPipeFileManager._append_landmarks_positions(
                file)
            self.save_to_file(raw_landmarks)

    @staticmethod
    def load_from_image(file_path, raw=True, height=None, width=None):
        data = MediaPipeFileManager._append_landmarks_positions(file_path)
        if not data:
            return None
        if raw:
            return data

        if not height or not width:
            raise Exception('You have to set height and width')
        scaled_data = []
        for x, y in data:
            scaled_data.append((int(x * width), int(y*height)))

        return scaled_data

    @staticmethod
    def _append_landmarks_positions(file_path):
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
                raw_landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    raw_landmarks.append((lm.x, lm.y))
                if raw_landmarks:
                    return raw_landmarks

    def save_to_file(self, data):
        with open(self.path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(data)

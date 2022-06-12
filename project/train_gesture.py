import cv2
import mediapipe as mp
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os
import glob
import joblib
from datetime import datetime
from argparse import ArgumentParser
import config
import shutil
import sys
from MediaPipeFileManager import MediaPipeFileManager


def capture_gesture(gesture_name):
    cam = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    path = f'{config.TRAINING_DATA_PATH}/{gesture_name}'
    if not os.path.isdir(path):
        os.mkdir(path)

    while True:
        success, img = cam.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        key = cv2.waitKey(1)
        test_size = len([name for name in os.listdir(
            path) if os.path.isfile(os.path.join(path, name))])

        if results.multi_hand_landmarks:
            if key % 256 == 32:
                for handLms in results.multi_hand_landmarks:
                    landmarks = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y*h)
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    img_path = f"{path}/gesture_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.png"

                    cv2.imwrite(img_path, img)
                    print(f"{img_path} written!")
        else:
            cv2.putText(img, "Nie wykryto dloni", (400, 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))

        cv2.putText(img, "Zarejestroj probke: SPACJA",
                    (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
        cv2.putText(img, f"Liczba zbioru testowego {test_size}", (
            0, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
        cv2.putText(img, "Aby wyjsc wcisnij: ESC", (0, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
        cv2.putText(img, "Usun ostatnia probke: x", (0, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))

        if key % 256 == 120:
            list_of_files = glob.glob(f'{path}/*.png')
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                os.remove(latest_file)
        if key % 256 == 27:
            cam.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow(f"Training gesture - {gesture_name}", img)


def delete_gesture(gesutre_name):
    path = f'{config.TRAINING_DATA_PATH}/{gesutre_name}'
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f'Gesutre {gesutre_name} deleted')
    else:
        print("Nie znaleziono gestu o podanej nazwie")


if __name__ == '__main__':
    parser = ArgumentParser()

    while True:
        print(" ")
        print("MENU: ")
        print('1 --- Zarejestruj nowy gest')
        print('2 --- Usuń gest')
        print('3 --- Trenuj')
        print('4 --- Wyjdź')
        print("")
        menu = int(input('Wybierz co chcesz zrobić: '))
        print("")

        if menu == 1:
            gesture = input('Wprowadź nazwę gestu: ')
            capture_gesture(gesture)
        elif menu == 2:
            gesture = input('Wprowadź nazwę gestu: ')
            delete_gesture(gesture)
        elif menu == 3:
            manager = MediaPipeFileManager()
            manager.export_images_to_csv(config.TRAINING_DATA_PATH)

            data = pd.read_csv(manager.path, delimiter=';')

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            neigh = KNeighborsClassifier(n_neighbors=3)
            clf = neigh.fit(X.values, y.values)
            joblib.dump(
                clf, f'{config.MODELS_FOLDER_PATH}/{config.MODEL_FILENAME}.pkl', compress=9)

            print('Pomyślnie wytrenowano model')
        elif menu == 4:
            sys.exit(0)

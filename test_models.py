import cv2
import mediapipe as mp
import time
import csv
import joblib
import numpy as np
from tensorflow.keras.models import load_model


def test_model(name):
    model = joblib.load(f'./data/models/{name}')

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

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                test_landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(test_landmarks)

                if prediction[0] == 1:
                    cv2.putText(img, 'gest', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Testing_models", img)
        cv2.waitKey(1)


def test_model2():
    model = load_model('./data/models/mp/mp_hand_gesture')
    f = open('./data/models/mp/gesture.names', 'r')
    class_names = f.read().split('\n')
    f.close()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        # Read each frame from the webcam
        _, frame = cap.read()
        x , y, c = frame.shape
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        # Show the final output
        if cv2.waitKey(1) == ord('q'):
                break
    
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        result = hands.process(framergb)
        className = ''
        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = class_names[classID]
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("Testing_models", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    test_model('one_class_model.pkl')
    # test_model2()



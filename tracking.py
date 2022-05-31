import cv2
import mediapipe as mp
import time
from ohbot import ohbot

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

success, img = cap.read()
rows, cols, _ = img.shape

x_medium = int(cols / 2)
x_center = int(cols / 2)
y_medium = int(rows / 2)
y_center = int(rows / 2)
x_position = 5 #środkowa pozycja
y_position = 4

print(x_medium, x_center)

# ohbot.move(0,x_position)
# ohbot.move(1,y_position)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                if id == 9:
                    if cx < x_center - 100:
                        if x_position >= 0.15:
                            x_position -= 0.15
                    elif cx > x_center + 100:
                        if x_position <= 9.45:
                            x_position += 0.15
                    print(f'cy {cy}, cx {cx}')
                    if cy < y_center - 100:
                        if y_position >= 0.15:
                            y_position -= 0.15
                    elif cy > y_center + 100:
                        if y_position <= 9.45:
                            y_position += 0.15

                print(y_position)
                # ohbot.move(1,y_position)
                # ohbot.move(0,x_position)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
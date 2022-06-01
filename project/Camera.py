
import cv2
import ohbot
from datetime import datetime
from MediaPipeFactory import MediaPipeHands

class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.x_position = 5
        self.y_position = 4
        self.img = self.cap.read()[1]

    def read_image(self):
        self.img = self.cap.read()[1]

    @property
    def imgRGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    @property
    def width(self):
        rows, cols, _ = self.img.shape
        return cols

    @property
    def height(self):
        rows, cols, _ = self.img.shape
        return rows

    @property
    def width_center(self):
        return int(self.width / 2)

    @property
    def height_center(self):
        return int(self.height / 2)

    def draw_circle(self, cx, cy):
        cv2.circle(self.img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

    def set_start_position(self):
        ohbot.move(2, self.x_position)
        ohbot.move(1, self.y_position)

    def fix_position(self, id, cx, cy):
        if id == MediaPipeHands.MIDDLE_FINGER_MCP:
            if cx < self.width_center - 100:
                if self.x_position > 0:
                    self.x_position -= 0.15
            elif cx > self.width_center + 100:
                if self.x_position < 10:
                    self.x_position += 0.15
            if cy < self.height_center - 100:
                if self.y_position >= 0.15:
                    self.y_position -= 0.15
            elif cy > self.height_center + 100:
                if self.y_position <= 9.45:
                    self.y_position += 0.15

        ohbot.wait(2)
        ohbot.move(0, self.camera.x_position, 2)
        ohbot.move(1, self.camera.y_position, 2)

    def show(self):
        cv2.imshow("Image", self.img)
        cv2.waitKey(1)

    def export_image(self, name, folder, subfolder):
        img_name = f"{name}.png"
        cv2.imwrite(f'../data/{folder}/{subfolder}/{img_name}', self.img)
        print(f"{img_name} written!")

    def put_text(self, text):
        cv2.putText(self.img, text, (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
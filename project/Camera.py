
import cv2
from ohbot import ohbot
from MediaPipeFactory import MediaPipeHands

class Camera():
    START_X_POS = 5
    START_Y_POS = 5

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.x_position = 5
        self.y_position = 4
        self.img = self.cap.read()[1]
        self.ohbot = ohbot

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
        print("CAMERA - start position")
        self.ohbot.move(0, self.START_X_POS, 1)
        self.ohbot.move(1, self.START_Y_POS, 1)

    def fix_position(self, id, cx, cy):
        norm_x = cx / self.width
        norm_y = cy / self.height

        norm_width_center = self.width_center / self.width
        norm_height_center = self.height_center / self.height

        if id == MediaPipeHands.MIDDLE_FINGER_MCP:
            if norm_x < norm_width_center - 0.15:
                if self.x_position > 0:
                    self.x_position -= 0.15
            elif norm_x > norm_width_center + 0.15:
                if self.x_position < 10:
                    self.x_position += 0.15
            if norm_y < norm_height_center - 0.15:
                if self.y_position >= 0.15:
                    self.y_position -= 0.15
            elif norm_y > norm_height_center + 0.15:
                if self.y_position <= 9.45:
                    self.y_position += 0.15

        self.ohbot.move(0, self.x_position, 1)
        self.ohbot.move(1, self.y_position, 1)

    def show(self):
        cv2.imshow("Image", self.img)
        cv2.waitKey(1)

    def export_image(self, name, folder, subfolder=None):
        img_name = f"{name}.png"
        if subfolder:
            cv2.imwrite(f'{folder}/{subfolder}/{img_name}', self.img)
        else:
            cv2.imwrite(f'{folder}/{img_name}', self.img)
        print(f"{img_name} written!")

    def put_text(self, text):
        cv2.putText(self.img, text, (10,70), cv2.FONT_ITALIC, 2, (255,0,255), 3)
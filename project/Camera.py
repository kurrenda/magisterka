
import cv2


class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.position = 5

    @property
    def img(self):
        return self.cap.read()

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
        return int(self.x_height / 2)

    @property
    def height_center(self):
        return int(self.y_height / 2)

    def draw_circle(self, cx, cy):
        cv2.circle(self.img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

    def show(self):
        cv2.imshow("Image", self.img)
        cv2.waitKey(1)
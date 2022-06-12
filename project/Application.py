from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import cv2
import config
from ComputerVision import ComputerVision


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Camera")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

        self.cv = ComputerVision()
        self.isLandmark = tk.BooleanVar()
        self.isLandmark.set(self.cv.mp_hands.draw_landmarks)
        self.isTracking = tk.BooleanVar()
        self.isTracking.set(self.cv.is_tracking)
        self.isCustom = tk.BooleanVar()
        self.isCustom.set(self.cv.is_custom_model)

        bottomframe = tk.Frame(self.root)
        bottomframe.pack(side=tk.BOTTOM)

        second_bottomframe = tk.Frame(self.root)
        second_bottomframe.pack(side=tk.BOTTOM, fill="x")

        btn_label = tk.Label(text="Ustaw gest", width=20)
        btn = tk.Button(self.root, text="Kciuk",
                        command=self.save_gesture_kciuk)
        btn1 = tk.Button(self.root, text="Otwarta dłoń",
                         command=self.save_gesture_otwarta)
        btn2 = tk.Button(self.root, text="Piesc",
                         command=self.save_gesture_piesc)
        c1 = tk.Checkbutton(bottomframe, text='Filtry sterowane gestami',
                            variable=self.isLandmark, command=self.set_landmarks)
        c2 = tk.Checkbutton(bottomframe, text='Włącz śledzenie ręki',
                            variable=self.isTracking, command=self.set_tracking)

        c3 = tk.Checkbutton(bottomframe, text='Wytrenowane gesty',
                            variable=self.isCustom, command=self.set_custom_gesture)

        btn_label.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        btn.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        btn1.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        btn2.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        c1.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        c2.pack(fill="both", side='left', expand=True, padx=10, pady=10)
        c3.pack(fill="both", side='left', expand=True, padx=10, pady=10)

        self.video_loop()

    def video_loop(self):
        self.cv.capture_frame()
        self.current_image = Image.fromarray(self.cv.camera.imgRGB)
        imgtk = ImageTk.PhotoImage(image=self.current_image)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
        self.root.after(15, self.video_loop)

    def save_gesture_kciuk(self):
        if self.cv.mp_hands.draw_landmarks:
            messagebox.showwarning(
                title='Zapis gestu - niepowodzenie', message="Aby zapisać gest, wyłącz filtry"
            )
            return
        self.cv.camera.export_image('kciuk', config.CLASSES_PHOTOS_PATH)
        self.cv.load_gestures_distances()

    def save_gesture_otwarta(self):
        if self.cv.mp_hands.draw_landmarks:
            messagebox.showwarning(
                title='Zapis gestu - niepowodzenie', message="Aby zapisać gest, wyłącz filtry"
            )
            return
        self.cv.camera.export_image('otwarta_reka', config.CLASSES_PHOTOS_PATH)
        self.cv.load_gestures_distances()

    def save_gesture_piesc(self):
        if self.cv.mp_hands.draw_landmarks:
            messagebox.showwarning(
                title='Zapis gestu - niepowodzenie', message="Aby zapisać gest, wyłącz filtry"
            )
            return
        self.cv.camera.export_image('piesc', config.CLASSES_PHOTOS_PATH)
        self.cv.load_gestures_distances()

    def set_landmarks(self):
        self.cv.is_background = False
        self.cv.is_cartoon = False
        self.cv.is_filter = self.isLandmark.get()
        self.cv.mp_hands.draw_landmarks = self.isLandmark.get()

    def set_tracking(self):
        if self.cv.camera.ohbot.connected:
            self.cv.is_tracking = self.isTracking.get()
        else:
            self.isTracking.set(False)
            messagebox.showwarning(
                title='Ohbot - niepowodzenie', message="Nie wykryto urządzenia ohbot"
            )
            return

    def set_custom_gesture(self):
        self.cv.is_custom_model = self.isCustom.get()

    def destructor(self):
        self.cv.camera.set_start_position()
        self.root.destroy()
        self.cv.camera.cap.release()
        cv2.destroyAllWindows()


pba = Application()
pba.root.mainloop()

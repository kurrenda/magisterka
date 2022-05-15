import cv2
import mediapipe as mp

face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

    image = cv2.imread('./face.jpg')
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
            cv2.imwrite('./face_result.' + '.png', annotated_image)

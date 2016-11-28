import cv2


class FaceDetection(object):
    def __init__(self, detector_path):
        self.face_cascade = cv2.CascadeClassifier(detector_path)

    def detect(self, img, scaleFactor=1.1, minNeighbors=5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detect = self.face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=(30, 30))
        return detect

    def detect_from_color(self, img, scaleFactor=1.1, minNeighbors=5):
        detect = self.face_cascade.detectMultiScale(img, scaleFactor, minNeighbors, minSize=(30, 30))
        return detect

    def display(self, img, scaleFactor=1.1, minNeighbors=5):
        faces = self.detect_from_color(img, scaleFactor, minNeighbors)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = img[y:y+h, x:x+w]

    def compute_proportions(self, faces):
        accumulate_width = 0
        accumulate_height = 0
        for (x, y, w, h) in faces:
            accumulate_width += w
            accumulate_height += h
        return accumulate_width, accumulate_height, len(faces)

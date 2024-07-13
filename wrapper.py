from tensorflow.keras.models import load_model
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import dlib
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import hashlib
from WideResNet import WideResNet


class WrapperClass:
    def __init__(self, img_size, depth, k, margin):
        self.img_size = img_size
        self.margin = margin
        self.ag_model = WideResNet(img_size, depth=depth, k=k)()
        self.ag_model.load_weights("models/weights.28-3.73.hdf5")
        self.emotion_model = load_model("models/emotion_little_vgg_2.h5")
        self.yolo = YOLO("yolov8s-oiv7.pt")
        self.ocr = PaddleOCR(
            use_angle_cls=True, lang="en", show_log=False
        )  
        self.emotion_classes = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }
        self.detector = dlib.get_frontal_face_detector()
        self.prev_frame = None

    def hash_frame(self, frame):
        frame_bytes = frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()

    def detection(self, frame):
        return self.detector(frame, 1)

    def preprocess_face(self, img_h, img_w, detected, faces, frame):
        preprocessed_faces_emo = []
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = (
                    d.left(),
                    d.top(),
                    d.right() + 1,
                    d.bottom() + 1,
                    d.width(),
                    d.height(),
                )
                xw1 = max(int(x1 - self.margin * w), 0)
                yw1 = max(int(y1 - self.margin * h), 0)
                xw2 = min(int(x2 + self.margin * w), img_w - 1)
                yw2 = min(int(y2 + self.margin * h), img_h - 1)
                faces[i, :, :, :] = cv2.resize(
                    frame[yw1 : yw2 + 1, xw1 : xw2 + 1, :],
                    (self.img_size, self.img_size),
                )
                face = frame[yw1 : yw2 + 1, xw1 : xw2 + 1, :]
                face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray_emo = cv2.resize(
                    face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA
                )
                face_gray_emo = face_gray_emo.astype("float") / 255.0
                face_gray_emo = img_to_array(face_gray_emo)
                face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                preprocessed_faces_emo.append(face_gray_emo)

        return preprocessed_faces_emo, faces

    def predict_ages(self, detected, faces):
        if len(detected) > 0:
            results = self.ag_model.predict(np.array(faces))
            predicted_genders = results[0]
            predicted_ages = np.argsort(results[1])[0][-5:]
            age_range = (min(predicted_ages), max(predicted_ages))

            return predicted_genders, age_range

        return None, None

    def predict_emo(self, detected, preprocessed_faces_emo):
        emo_labels = []
        for i, d in enumerate(detected):
            preds = self.emotion_model.predict(preprocessed_faces_emo[i])[0]
            e = self.emotion_classes[preds.argmax()]
            emo_labels.append(e)
        return emo_labels

    def yolo_pred(self, frame):
        res = self.yolo(frame)
        return [(i["name"], i["box"]) for j in res for i in j.summary()]

    def vision_pred(self, frame):
        preprocessed_faces_emo = []
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)
        detected = self.detector(frame, 1)
        faces = np.empty((len(detected), self.img_size, self.img_size, 3))
        preprocessed_faces_emo, faces = self.preprocess_face(
            img_h, img_w, detected, faces, frame
        )
        gender, age = self.predict_ages(detected, faces)
        emo_labels = self.predict_emo(detected, preprocessed_faces_emo)
        print(gender, age, emo_labels)
        result = self.yolo_pred(frame)
        print(result)

    def ocr_pred(self, frame):
        frame_hash = self.hash_frame(frame)
        if frame_hash == self.prev_frame:
            print("skipping frame")
            return
        result = self.ocr.ocr(frame, cls=True)
        detected_text = ""
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    detected_text += " " + line[1][0]
        # corrected_text = self.correct_text(detected_text)
        print(detected_text)
        self.prev_frame = frame_hash

    def make_prediction(self, video_path):
        a = time.time()
        # output_video_path = "output.mp4"
        cap = cv2.VideoCapture(video_path)
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) // 2
        frame_count = 0
        print(fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (fps) != 0:
                frame_count += 1
                continue

            print(frame_count)
            self.ocr_pred(frame)
            self.vision_pred(frame)
            frame_count += 1

        cap.release()
        # cv2.destroyAllWindows()
        b = time.time()
        print(b - a)

    def __call__(
        self,
        video_path,
    ):
        self.make_prediction(video_path)


model = WrapperClass(64, 16, 8, 0.4)
model("/home/shusrith/vid.mp4")

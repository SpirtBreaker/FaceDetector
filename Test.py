import cv2
from fer import FER

class FaceDetector:
    def __init__(self):
        self.emo_detector = FER(mtcnn=True)  # Создаем объект FER

    def highlightFace(self, net, frame, conf_threshold=0.95):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        emotions = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                face_roi = frameOpencvDnn[y1:y2, x1:x2]
                dominant_emotion, emotion_score = self.emo_detector.top_emotion(face_roi)
                emotions.append(dominant_emotion)
                color = (255, 255, 255, 255)
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
                cv2.putText(frameOpencvDnn, f"{str(emotion_score)} {dominant_emotion}",
                            (x1, y1 - 15),
                            cv2.FONT_ITALIC, 0.5, color, 1, cv2.LINE_AA, )
        return frameOpencvDnn, faceBoxes, emotions

    def detect_faces_and_emotions(self, frame):
        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        resultImg, faceBoxes, emotions = self.highlightFace(faceNet, frame)
        return faceBoxes, emotions

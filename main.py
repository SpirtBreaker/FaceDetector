import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from fer import FER

class FaceDetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Детектор лиц и эмоций")

        # Создаем элементы управления
        self.video_label = tk.Label(master)
        self.video_label.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.emotions_text = tk.Text(master, height=10, width=30)
        self.emotions_text.pack()

        # Загружаем классификатор для распознавания лиц
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Создаем объект FER для распознавания эмоций
        self.emo_detector = FER(mtcnn=True)

        # Запускаем поток для обработки видео
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.start()

        # Время последнего обновления эмоций
        self.last_update_time = time.time()

    def process_video(self):
        # Открываем камеру
        cap = cv2.VideoCapture(0)

        while True:
            # Захватываем кадр с камеры
            ret, frame = cap.read()

            if ret:
                # Корректируем баланс белого
                frame = self.adjust_white_balance(frame)

                # Обнаруживаем лица и эмоции на кадре
                faces, emotions = self.detect_faces_and_emotions(frame)

                # Проверяем, прошло ли достаточно времени для обновления эмоций
                current_time = time.time()
                if current_time - self.last_update_time >= 1:  # Обновляем эмоции каждую секунду
                    self.update_gui(frame, len(faces), emotions)
                    self.last_update_time = current_time

            # Задержка для ограничения частоты кадров
            cv2.waitKey(30)

        cap.release()

    def adjust_white_balance(self, frame):
        # Конвертируем изображение в цветовое пространство LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Разделяем каналы LAB
        l, a, b = cv2.split(lab)

        # Вычисляем среднее значение для канала L
        l_mean = cv2.mean(l)[0]

        # Вычисляем коэффициенты усиления для каналов A и B
        a_gain = 128.0 / l_mean
        b_gain = 128.0 / l_mean

        # Применяем коэффициенты усиления к каналам A и B
        a = cv2.multiply(a, a_gain)
        b = cv2.multiply(b, b_gain)

        # Объединяем каналы LAB обратно
        lab = cv2.merge((l, a, b))

        # Конвертируем изображение обратно в цветовое пространство BGR
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def detect_faces_and_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        emotions = []

        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            dominant_emotion, emotion_score = self.emo_detector.top_emotion(roi_color)
            emotions.append(self.get_emotion_text(dominant_emotion))

        return faces, emotions

    def get_emotion_text(self, emotion):
        emotions = {
            'angry': 'злость',
            'disgust': 'отвращение',
            'fear': 'страх',
            'happy': 'радость',
            'sad': 'грусть',
            'surprise': 'удивление',
            'neutral': 'нейтральное'
        }
        return emotions.get(emotion, 'неизвестно')

    def update_gui(self, frame, num_faces, emotions):
        # Конвертируем изображение в формат, подходящий для Tkinter
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Обновляем элементы управления
        self.video_label.configure(image=image)
        self.video_label.image = image
        self.result_label.config(text=f"Найдено {num_faces} лиц")
        self.emotions_text.delete('1.0', tk.END)  # Очищаем текстовое поле
        for i, emotion in enumerate(emotions, start=1):
            self.emotions_text.insert(tk.END, f"Лицо {i}: {emotion}\n")


root = tk.Tk()
gui = FaceDetectionGUI(root)
root.mainloop()
from keras.src.saving.saving_api import load_model
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import pyttsx3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.filechooser import FileChooserListView

class SignLanguageSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.dataset_dir = "dataset"
        self.model_path = "gesture_model.h5"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.gestures = self.get_existing_gestures()

        # Inicialización del motor de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 100)  # Velocidad del habla
        self.engine.setProperty('volume', 1)  # Volumen del habla

        self.last_prediction = None  # Para rastrear la última predicción

    def get_existing_gestures(self):
        if not os.path.exists(self.dataset_dir):
            return []
        return os.listdir(self.dataset_dir)

    def detect_hands(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Detección de Manos", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def collect_data(self, gesture, num_frames):
        save_dir = os.path.join(self.dataset_dir, gesture)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        counter = 0

        while counter < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ]
                    landmarks = np.array(landmarks).flatten()
                    np.save(os.path.join(save_dir, f"frame_{counter}.npy"), landmarks)
                    counter += 1

            if counter >= num_frames:
                break

        cap.release()
        self.gestures = self.get_existing_gestures()

    def train_model(self):
        if not self.gestures:
            return "No hay datos para entrenar"

        X, y, self.gestures = self.load_data()
        y = tf.keras.utils.to_categorical(y)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.gestures), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
        model.save(self.model_path)
        return f"Modelo guardado en {self.model_path}"

    def load_data(self):
        data, labels, gestures = [], [], []
        for label, gesture in enumerate(os.listdir(self.dataset_dir)):
            gesture_dir = os.path.join(self.dataset_dir, gesture)
            gestures.append(gesture)
            for file in os.listdir(gesture_dir):
                filepath = os.path.join(gesture_dir, file)
                landmarks = np.load(filepath)
                data.append(landmarks)
                labels.append(label)
        return np.array(data), np.array(labels), gestures

    def evaluate(self):
        if not os.path.exists(self.model_path):
            return "Modelo no encontrado"

        model = load_model(self.model_path)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ]).flatten().reshape(1, -1)

                    prediction = model.predict(landmarks, verbose=0)
                    gesture_index = np.argmax(prediction)
                    gesture_name = self.gestures[gesture_index]
                    confidence = prediction[0][gesture_index]

                    if confidence > 0.7 and gesture_name != self.last_prediction:
                        self.engine.say(gesture_name)
                        self.engine.runAndWait()
                        self.last_prediction = gesture_name

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()

class SignLanguageApp(App):
    def build(self):
        self.system = SignLanguageSystem()
        layout = BoxLayout(orientation='vertical')

        self.label = Label(text="Bienvenido al sistema de lenguaje de señas")
        layout.add_widget(self.label)

        detect_button = Button(text="Detectar manos")
        detect_button.bind(on_press=lambda x: self.run_detection())
        layout.add_widget(detect_button)

        collect_button = Button(text="Recolectar datos")
        collect_button.bind(on_press=lambda x: self.run_collection())
        layout.add_widget(collect_button)

        train_button = Button(text="Entrenar modelo")
        train_button.bind(on_press=lambda x: self.train_model())
        layout.add_widget(train_button)

        evaluate_button = Button(text="Evaluar")
        evaluate_button.bind(on_press=lambda x: self.evaluate())
        layout.add_widget(evaluate_button)

        return layout

    def run_detection(self):
        self.system.detect_hands()
        self.label.text = "Detección de manos completada"

    def run_collection(self):
        gesture = "GestoEjemplo"  # Esto debería venir de un campo de entrada
        num_frames = 50  # Ajustar como sea necesario
        self.system.collect_data(gesture, num_frames)
        self.label.text = f"Datos recolectados para: {gesture}"

    def train_model(self):
        message = self.system.train_model()
        self.label.text = message

    def evaluate(self):
        self.system.evaluate()
        self.label.text = "Evaluación completada"

if __name__ == "__main__":
    SignLanguageApp().run()

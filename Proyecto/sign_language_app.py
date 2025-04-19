import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import pyttsx3

# Kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

# Configuration files
model_file = "hand_gesture_nn_model_5_2.h5"
scaler_file = "hand_gesture_scaler_5_2.pkl"
encoder_file = "hand_gesture_encoder_5_2.pkl"
gesture_data = "gesture_data_5_2.pkl"
data_dir = "hand_gestures_data_4_3"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_lock = threading.Lock()
last_spoken_gesture = None

class UDPCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # For testing, use local camera
        
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()

def speak_text(text):
    global last_spoken_gesture
    with tts_lock:
        if text != last_spoken_gesture:
            last_spoken_gesture = text
            tts_engine.say(text)
            tts_engine.runAndWait()

def extract_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    landmarks_data = []
    hands_detected = False
    
    if results.multi_hand_landmarks:
        hands_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_data.extend(landmarks)
    
    while len(landmarks_data) < 21 * 3 * 2:
        landmarks_data.append(0.0)
    
    landmarks_data = landmarks_data[:21 * 3 * 2]
    return landmarks_data, hands_detected

class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, input_data):
        input_data = np.array(input_data, dtype=self.input_details[0]['dtype'])
        if len(input_data.shape) == len(self.input_details[0]['shape']) - 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

class SignLanguageUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 20

        # Estado de la aplicación
        self.is_collecting = False
        self.is_trained = False
        self.evaluation_mode = False
        self.samples_collected = 0
        self.current_gesture = ""

        # Crear widgets
        self.status_label = Label(
            text='Estado: Modelo NO ENTRENADO | Datos: 0 muestras',
            size_hint_y=0.1
        )

        # Área de la cámara
        self.image = Image(size_hint_y=0.6)

        # Botones del menú
        buttons_layout = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=10)
        self.collect_btn = Button(
            text='1. Recolectar nueva seña',
            on_press=self.start_collection_popup
        )
        self.train_btn = Button(
            text='2. Entrenar modelo',
            on_press=self.train_model
        )
        self.evaluate_btn = Button(
            text='3. Evaluar en tiempo real',
            on_press=self.start_evaluation
        )
        self.exit_btn = Button(
            text='4. Salir',
            on_press=self.stop_app
        )

        # Agregar botones al layout
        buttons_layout.add_widget(self.collect_btn)
        buttons_layout.add_widget(self.train_btn)
        buttons_layout.add_widget(self.evaluate_btn)
        buttons_layout.add_widget(self.exit_btn)

        # Agregar widgets al layout principal
        self.add_widget(self.status_label)
        self.add_widget(self.image)
        self.add_widget(buttons_layout)

        # Inicializar cámara y modelo
        self.setup_camera()
        self.load_model_and_data()

        # Iniciar actualización de cámara
        Clock.schedule_interval(self.update, 1.0/30.0)

    def setup_camera(self):
        try:
            self.cap = UDPCamera()
            print("Cámara inicializada correctamente")
        except Exception as e:
            print(f"Error al iniciar la cámara: {str(e)}")

    def load_model_and_data(self):
        if os.path.exists("modelo_optimizadotl.tflite"):
            self.tflite_model = TFLiteModel("modelo_optimizadotl.tflite")
            self.is_trained = True
        self.update_status()

    def update_status(self):
        self.status_label.text = f"Estado: Modelo {'ENTRENADO' if self.is_trained else 'NO ENTRENADO'}"

    def start_collection_popup(self, instance):
        content = BoxLayout(orientation='vertical')
        self.gesture_input = TextInput(multiline=False)
        content.add_widget(Label(text='Ingrese nombre de la seña:'))
        content.add_widget(self.gesture_input)
        content.add_widget(Button(text='Iniciar', on_press=self.start_collection))

        self.popup = Popup(
            title='Nueva Seña',
            content=content,
            size_hint=(None, None), size=(400, 200)
        )
        self.popup.open()

    def start_collection(self, instance):
        gesture_name = self.gesture_input.text
        if gesture_name:
            self.current_gesture = gesture_name
            self.is_collecting = True
            self.samples_collected = 0
            self.popup.dismiss()

    def train_model(self, instance):
        if not self.is_trained:
            Popup(title='Error',
                  content=Label(text='El modelo aún no está entrenado'),
                  size_hint=(None, None), size=(300, 150)).open()
        else:
            self.update_status()

    def start_evaluation(self, instance):
        if self.is_trained:
            self.evaluation_mode = True
        else:
            Popup(title='Error',
                  content=Label(text='¡Entrena el modelo primero!'),
                  size_hint=(None, None), size=(300, 150)).open()

    def stop_app(self, instance):
        App.get_running_app().stop()

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Procesar frame
        landmarks, hands_detected = extract_hand_landmarks(frame)

        # Lógica según el modo actual
        if self.is_collecting:
            self.handle_collection_mode(frame, landmarks, hands_detected)
        elif self.evaluation_mode:
            self.handle_evaluation_mode(frame, landmarks, hands_detected)

        # Convertir frame para Kivy
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def handle_collection_mode(self, frame, landmarks, hands_detected):
        if hands_detected:
            self.samples_collected += 1
            if self.samples_collected >= 100:  # Example limit
                self.is_collecting = False
                self.update_status()

    def handle_evaluation_mode(self, frame, landmarks, hands_detected):
        if hands_detected:
            prediction = self.tflite_model.predict(landmarks)
            confidence = np.max(prediction)
            if confidence > 0.95:
                prediction_text = "Detectada seña"  # You'll need to map this to actual labels
                threading.Thread(target=speak_text, args=(prediction_text,), daemon=True).start()

class SignLanguageApp(App):
    def build(self):
        return SignLanguageUI()

if __name__ == "__main__":
    SignLanguageApp().run()
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import threading



# Importa las funciones necesarias de tu código original
from core.detector import (
    detect_hands, 
    collect_data, 
    train_model, 
    evaluate, 
    convert_to_tflite,
    process_frame
)

class SignLanguageApp(App): 
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Panel de cámara
        self.camera_preview = Label(size_hint=(1, 0.7))
        self.layout.add_widget(self.camera_preview)
        
        # Panel de controles
        controls = BoxLayout(size_hint=(1, 0.3), orientation='vertical')
        
        self.status_label = Label(text="Estado: Listo", size_hint=(1, 0.2))
        controls.add_widget(self.status_label)
        
        btn_layout = BoxLayout(size_hint=(1, 0.8))
        btn_layout.add_widget(Button(text='Detectar', on_press=self.start_detection))
        btn_layout.add_widget(Button(text='Entrenar', on_press=self.start_training))
        btn_layout.add_widget(Button(text='Traducir', on_press=self.start_translation))
        controls.add_widget(btn_layout)
        
        self.layout.add_widget(controls)
        return self.layout

    def start_detection(self, instance):
        self.status_label.text = "Estado: Detección activa"
        self.stop_event = threading.Event()
        threading.Thread(
            target=detect_hands,
            args=(self.update_frame, self.stop_event),
            daemon=True
        ).start()

    def update_frame(self, frame):
        try:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), 
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_preview.texture = texture
        except:
            pass

    def start_training(self, instance):
        self.status_label.text = "Estado: Entrenando modelo..."
        threading.Thread(target=train_model).start()

    def start_translation(self, instance):
        self.status_label.text = "Estado: Traduciendo..."
        threading.Thread(target=evaluate_kivy, args=(self.update_prediction,)).start()

    def update_frame(self, frame):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self.camera_preview.texture = texture

    def update_prediction(self, gesture, confidence):
        self.status_label.text = f"Predicción: {gesture} ({confidence:.0%})"

def detect_hands_kivy(callback):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame)
            callback(cv2.flip(frame, 0))
        else:
            break
    cap.release()

def evaluate_kivy(callback):
    # Implementación similar a tu función evaluate() original pero con callback
    pass

if __name__ == '__main__':
    SignLanguageApp().run()
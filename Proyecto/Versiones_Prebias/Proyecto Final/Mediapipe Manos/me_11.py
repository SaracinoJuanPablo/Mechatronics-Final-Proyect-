from keras.src.saving.saving_api import load_model
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import pyttsx3
import time

class SignLanguageSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=2, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.dataset_dir = "dataset"
        self.model_path = "gesture_model.h5"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.gestures = self.get_existing_gestures()
        
        # Inicialización del motor de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Velocidad del habla
        self.engine.setProperty('volume', 1)  # Volumen del habla

        self.last_prediction = None  # Para rastrear la última predicción

        # Optimización de GPU si está disponible
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def get_existing_gestures(self):
        if not os.path.exists(self.dataset_dir):
            return []
        return os.listdir(self.dataset_dir)

    def evaluate(self):
        if not os.path.exists(self.model_path):
            print("\nNo se encontró el modelo entrenado. Por favor, primero entrene el modelo.")
            return

        print("\nCargando modelo y optimizando evaluación...")
        model = load_model(self.model_path)

        cap = cv2.VideoCapture(0)
        last_prediction_time = 0
        prediction_interval = 0.5  # 500 ms entre predicciones

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).flatten()
                    landmarks = np.expand_dims(landmarks, axis=0)

                    current_time = time.time()
                    if current_time - last_prediction_time > prediction_interval:
                        prediction = model.predict(landmarks, verbose=0)
                        gesture_index = np.argmax(prediction)
                        confidence = prediction[0][gesture_index]
                        gesture_name = self.gestures[gesture_index]

                        if confidence > 0.8 and gesture_name != self.last_prediction:
                            self.engine.say(gesture_name)
                            self.engine.runAndWait()
                            self.last_prediction = gesture_name

                        last_prediction_time = current_time

                        cv2.putText(frame, f"Gesto: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Evaluación de Señas", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    system = SignLanguageSystem()

    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Evaluar Gestos")
        print("2. Salir")

        choice = input("\nSeleccione una opción: ")

        if choice == '1':
            system.evaluate()
        elif choice == '2':
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()

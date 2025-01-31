"""import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import pyttsx3
import time
from threading import Thread, Lock

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
        # Configurar la voz en español si está disponible
        spanish_voice = None
        for voice in self.engine.getProperty('voices'):
            if "spanish" in voice.languages[0].lower():
                spanish_voice = voice.id
                break
        if spanish_voice:
            self.engine.setProperty('voice', spanish_voice)
        
        # Variables para control de voz
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.speak_lock = Lock()
        
    def speak_text(self, text):
        #Función para reproducir texto por voz en un hilo separado
        def speak_thread():
            with self.speak_lock:
                self.engine.say(text)
                self.engine.runAndWait()
        
        Thread(target=speak_thread).start()

    # [Mantener todos los métodos anteriores sin cambios hasta evaluate()]

    def evaluate(self):
        if not os.path.exists(self.model_path):
            print("\nNo se encontró el modelo entrenado. Por favor, primero entrene el modelo.")
            return

        print("\nCargando modelo y iniciando evaluación...")
        model = load_model(self.model_path)
        
        cap = cv2.VideoCapture(0)
        print("\nMostrando predicciones en tiempo real. Presiona 'ESC' para salir.")

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

                    prediction = model.predict(landmarks, verbose=0)
                    gesture_index = np.argmax(prediction)
                    confidence = prediction[0][gesture_index]
                    gesture_name = self.gestures[gesture_index]

                    # Mostrar predicción en pantalla
                    cv2.putText(frame, f"Gesto: {gesture_name}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Reproducir por voz si ha pasado el tiempo suficiente y es un gesto diferente
                    current_time = time.time()
                    if (current_time - self.last_spoken_time > 3 and 
                        gesture_name != self.last_spoken and 
                        confidence > 0.7):  # Solo reproduce si la confianza es mayor al 70%
                        self.speak_text(gesture_name)
                        self.last_spoken = gesture_name
                        self.last_spoken_time = current_time

            cv2.imshow("Evaluación de Señas", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # [Mantener los métodos retrain_gesture() y el resto sin cambios]

def main():
    try:
        import pyttsx3
    except ImportError:
        print("El módulo pyttsx3 no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(["pip", "install", "pyttsx3"])
        print("pyttsx3 instalado correctamente.")
    
    system = SignLanguageSystem()
    
    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Detectar Manos")
        print("2. Recolectar Datos")
        print("3. Entrenar Modelo")
        print("4. Evaluar (con reproducción de voz)")
        print("5. Reentrenar Gesto")
        print("6. Salir")
        
        choice = input("\nSeleccione una opción: ")
        
        if choice == '1':
            system.detect_hands()
        elif choice == '2':
            system.collect_data()
        elif choice == '3':
            system.train_model()
        elif choice == '4':
            system.evaluate()
        elif choice == '5':
            system.retrain_gesture()
        elif choice == '6':
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import pyttsx3

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
        self.engine.setProperty('rate', 150)  # Velocidad del habla
        self.engine.setProperty('volume', 1)  # Volumen del habla

    def get_existing_gestures(self):
        if not os.path.exists(self.dataset_dir):
            return []
        return os.listdir(self.dataset_dir)

    def detect_hands(self):
        print("\nIniciando detección de manos. Presiona 'ESC' para salir.")
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

    def collect_data(self):
        gesture = input("\nIngrese la palabra o letra para la cual desea recolectar datos: ").upper()
        num_frames = int(input("Ingrese el número de frames a capturar (recomendado: 100): "))
        
        save_dir = os.path.join(self.dataset_dir, gesture)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nRecolectando datos para el gesto '{gesture}'. Presiona 'ESC' para cancelar.")
        print("Mantenga la seña frente a la cámara...")
        
        cap = cv2.VideoCapture(0)
        counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    
                    landmarks = np.array(landmarks).flatten()
                    np.save(os.path.join(save_dir, f"frame_{counter}.npy"), landmarks)
                    counter += 1
                    
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f"Frames capturados: {counter}/{num_frames}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Recolección de Datos", frame)
            if cv2.waitKey(1) & 0xFF == 27 or counter >= num_frames:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.gestures = self.get_existing_gestures()
        print(f"\nSe recolectaron {counter} frames para el gesto '{gesture}'")

    def train_model(self):
        if not self.gestures:
            print("\nNo hay datos recolectados. Por favor, primero recolecte datos de gestos.")
            return

        print("\nCargando datos y preparando el entrenamiento...")
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
        
        print("\nIniciando entrenamiento...")
        history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
        model.save(self.model_path)
        print(f"\nModelo guardado en {self.model_path}")
        
        # Mostrar métricas finales
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"Precisión de validación final: {val_accuracy:.2%}")

    def load_data(self):
        data = []
        labels = []
        gestures = []
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
            print("\nNo se encontró el modelo entrenado. Por favor, primero entrene el modelo.")
            return

        print("\nCargando modelo y iniciando evaluación...")
        model = load_model(self.model_path)
        
        cap = cv2.VideoCapture(0)
        print("\nMostrando predicciones en tiempo real. Presiona 'ESC' para salir.")

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

                    prediction = model.predict(landmarks, verbose=0)
                    gesture_index = np.argmax(prediction)
                    confidence = prediction[0][gesture_index]
                    gesture_name = self.gestures[gesture_index]

                    cv2.putText(frame, f"Gesto: {gesture_name}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Reproducir el nombre del gesto si la confianza es mayor al 80%
                    if confidence > 0.8:
                        self.engine.say(gesture_name)
                        self.engine.runAndWait()

            cv2.imshow("Evaluación de Señas", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def retrain_gesture(self):
        if not self.gestures:
            print("\nNo hay gestos para reentrenar. Primero recolecte datos.")
            return

        print("\nGestos disponibles para reentrenar:")
        for i, gesture in enumerate(self.gestures):
            print(f"{i+1}. {gesture}")

        try:
            choice = int(input("\nSeleccione el número del gesto a reentrenar: ")) - 1
            if 0 <= choice < len(self.gestures):
                gesture = self.gestures[choice]
                gesture_dir = os.path.join(self.dataset_dir, gesture)
                
                # Eliminar datos anteriores
                for file in os.listdir(gesture_dir):
                    os.remove(os.path.join(gesture_dir, file))
                
                print(f"\nDatos anteriores de '{gesture}' eliminados.")
                self.collect_data()  # Recolectar nuevos datos
                self.train_model()   # Reentrenar el modelo
            else:
                print("\nSelección inválida.")
        except ValueError:
            print("\nPor favor, ingrese un número válido.")

def main():
    system = SignLanguageSystem()
    
    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Detectar Manos")
        print("2. Recolectar Datos")
        print("3. Entrenar Modelo")
        print("4. Evaluar")
        print("5. Reentrenar Gesto")
        print("6. Salir")
        
        choice = input("\nSeleccione una opción: ")
        
        if choice == '1':
            system.detect_hands()
        elif choice == '2':
            system.collect_data()
        elif choice == '3':
            system.train_model()
        elif choice == '4':
            system.evaluate()
        elif choice == '5':
            system.retrain_gesture()
        elif choice == '6':
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()
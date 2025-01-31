import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os
from time import sleep

class SignLanguageSystem:
    def __init__(self):
        # Inicialización de MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configuración de detectores
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configuración de Text-to-Speech
        self.engine = pyttsx3.init()
        
        # Configuración de directorios y archivos
        self.data_dir = "sign_language_data"
        self.model_file = "sign_language_model.h5"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuración de landmarks
        self.n_pose_landmarks = 33 * 3  # 33 puntos con x, y, z
        self.n_hand_landmarks = 21 * 3  # 21 puntos con x, y, z
        self.total_landmarks = self.n_pose_landmarks + (self.n_hand_landmarks * 2)  # Pose + 2 manos
        
    def process_frame(self, frame):
        """Procesa un frame y retorna los resultados de pose y manos"""
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        return pose_results, hands_results
    
    def extract_landmarks(self, pose_results, hands_results):
        """Extrae y normaliza los landmarks de pose y manos"""
        landmarks = []
        
        # Extraer landmarks de pose
        if pose_results.pose_landmarks:
            pose_landmarks = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
            landmarks.extend(np.array(pose_landmarks).flatten())
        else:
            landmarks.extend([0] * self.n_pose_landmarks)
        
        # Extraer landmarks de manos (hasta 2 manos)
        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hand_landmarks_list.extend(np.array(hand_points).flatten())
        
        # Rellenar con ceros si faltan manos
        while len(hand_landmarks_list) < self.n_hand_landmarks * 2:
            hand_landmarks_list.extend([0] * self.n_hand_landmarks)
        
        landmarks.extend(hand_landmarks_list)
        return np.array(landmarks)
    
    def collect_data(self, sign_name):
        """Recolecta datos para una seña específica"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        count = 0
        total_samples = int(input("Número de muestras a recolectar: "))
        
        print(f"Recolectando datos para: {sign_name}")
        print("Presione ESC para cancelar")
        
        while count < total_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame")
                break
            
            pose_results, hands_results = self.process_frame(frame)
            
            # Extraer landmarks
            landmarks = self.extract_landmarks(pose_results, hands_results)
            
            # Guardar landmarks
            if len(landmarks) > 0:
                np.save(os.path.join(sign_dir, f"sample_{count}.npy"), landmarks)
                count += 1
            
            # Dibujar landmarks
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Mostrar progreso
            cv2.putText(frame, f"Muestras: {count}/{total_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Recolección de Datos", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Recolección completada: {count} muestras guardadas")
    
    def train_model(self):
        """Entrena el modelo con los datos recolectados"""
        if not os.listdir(self.data_dir):
            print("No hay datos para entrenar")
            return
        
        X = []
        y = []
        class_names = sorted(os.listdir(self.data_dir))
        
        # Cargar datos
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for sample_file in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_file)
                landmarks = np.load(sample_path)
                X.append(landmarks)
                y.append(class_idx)
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y)
        
        # Crear modelo
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.total_landmarks,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(class_names), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Entrenar modelo
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Guardar modelo
        model.save(self.model_file)
        print("Modelo entrenado y guardado correctamente")
    
    def evaluate(self):
        """Evalúa el modelo en tiempo real"""
        if not os.path.exists(self.model_file):
            print("No se encontró el modelo entrenado")
            return
        
        model = tf.keras.models.load_model(self.model_file)
        class_names = sorted(os.listdir(self.data_dir))
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        last_prediction = ""
        last_prediction_time = 0
        
        print("Iniciando evaluación. Presione ESC para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_results, hands_results = self.process_frame(frame)
            landmarks = self.extract_landmarks(pose_results, hands_results)
            
            # Realizar predicción
            prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Dibujar landmarks
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Mostrar predicción
            cv2.putText(frame, f"Seña: {predicted_class}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Reproducir predicción por voz
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if (confidence > 0.7 and 
                predicted_class != last_prediction and 
                current_time - last_prediction_time > 3):
                self.engine.say(predicted_class)
                self.engine.runAndWait()
                last_prediction = predicted_class
                last_prediction_time = current_time
            
            cv2.imshow("Evaluación", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = SignLanguageSystem()
    
    while True:
        print("\n=== Sistema de Reconocimiento de Lenguaje de Señas ===")
        print("1. Ver detección de pose y manos")
        print("2. Recolectar datos de señas")
        print("3. Entrenar modelo")
        print("4. Evaluar en tiempo real")
        print("5. Salir")
        
        option = input("\nSeleccione una opción: ")
        
        if option == "1":
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                pose_results, hands_results = system.process_frame(frame)
                
                if pose_results.pose_landmarks:
                    system.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, 
                                               system.mp_pose.POSE_CONNECTIONS)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        system.mp_draw.draw_landmarks(frame, hand_landmarks,
                                                   system.mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow("Detección", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif option == "2":
            sign_name = input("Nombre de la seña a recolectar: ")
            system.collect_data(sign_name)
        
        elif option == "3":
            system.train_model()
        
        elif option == "4":
            system.evaluate()
        
        elif option == "5":
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
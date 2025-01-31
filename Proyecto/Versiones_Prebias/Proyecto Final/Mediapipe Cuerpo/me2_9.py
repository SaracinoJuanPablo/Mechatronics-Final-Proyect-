import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os
from time import sleep 
import time

# Ahora puedes usar time.time()
current_time = time.time()

class SignLanguageSystem:
    def __init__(self):
        # Inicialización de MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Reducir la complejidad
        )

        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
            model_complexity=0  # Complejidad mínima
        )

        
        self.engine = pyttsx3.init()
        self.data_dir = "sign_language_data"
        self.model_file = "sign_language_model.h5"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuración para secuencias de movimiento
        self.sequence_length = 30  # Frames por secuencia
        self.n_pose_landmarks = 33 * 3
        self.n_hand_landmarks = 21 * 3
        self.total_landmarks = self.n_pose_landmarks + (self.n_hand_landmarks * 2)

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
        """Recolecta secuencias de movimiento para una seña específica"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Cambiar el índice para usar DroidCam
        #cap = cv2.VideoCapture(1)  # Usa el índice correspondiente de DroidCam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)  # Ajusta a 30 FPS
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 640, 480)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        total_sequences = int(input("Número de secuencias a recolectar (recomendado: 20-30): "))

        print("\nInstrucciones:")
        print(f"1. Cada secuencia grabará {self.sequence_length} frames de movimiento")
        print("2. Presiona Enter para comenzar la grabación de las secuencias")
        print("3. Realiza el movimiento completo de la seña")
        print("4. La grabación se detendrá automáticamente")
        print("5. Presiona ESC para cancelar")

        sequence_count = 0
        frame_count = 0
        is_recording = False
        current_sequence = []

        frame_skip = 2  # Procesar un frame de cada 2
        frame_counter = 0

        input("Presiona Enter para iniciar la grabación de secuencias.")

        while sequence_count < total_sequences:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % frame_skip != 0:  # Salta este frame
                continue

            frame = cv2.flip(frame, 1)
            pose_results, hands_results = self.process_frame(frame)  
            # Dibujar landmarks
            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, 
                                        self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)

            # Mostrar estado
            if not is_recording:
                is_recording = True
                current_sequence = []
                frame_count = 0
                print(f"\nGrabando secuencia {sequence_count + 1}...")

            if is_recording:
                cv2.putText(frame, f"Grabando secuencia... Frame {frame_count}/{self.sequence_length}", 
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Capturar frame para la secuencia
                landmarks = self.extract_landmarks(pose_results, hands_results)
                current_sequence.append(landmarks)
                frame_count += 1

                # Verificar si la secuencia está completa
                if frame_count >= self.sequence_length:
                    # Guardar secuencia
                    sequence_data = np.array(current_sequence)
                    np.save(os.path.join(sign_dir, f"sequence_{sequence_count}.npy"), 
                            sequence_data)
                    print(f"Secuencia {sequence_count + 1}/{total_sequences} guardada")

                    # Resetear para la siguiente secuencia
                    sequence_count += 1
                    frame_count = 0
                    is_recording = False
                    current_sequence = []

            # Mostrar contador de secuencias
            cv2.putText(frame, f"Secuencias: {sequence_count}/{total_sequences}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Recolección de Datos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC - Salir
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal de secuencias guardadas: {sequence_count}")
        print(f"Ubicación: {sign_dir}")

# Otras funciones se mantienen igual...

# Punto de entrada principal del programa
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

        if option == "2":
            sign_name = input("Nombre de la seña a recolectar: ")
            system.collect_data(sign_name)
        # Implementar otras opciones aquí...

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os
from time import time
from collections import deque
from threading import Thread

class SignLanguageSystem:
    def __init__(self):
        # Inicialización de MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

        self.engine = pyttsx3.init()
        self.data_dir = "sign_language_data"
        self.model_file = "sign_language_model.h5"
        os.makedirs(self.data_dir, exist_ok=True)

        # Configuración para secuencias de movimiento
        self.sequence_length = 30
        self.n_pose_landmarks = 33 * 3
        self.n_hand_landmarks = 21 * 3
        self.total_landmarks = self.n_pose_landmarks + (self.n_hand_landmarks * 2)

    def process_frame(self, frame):
        """Procesa un frame y retorna los resultados de pose y manos"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        return pose_results, hands_results

    def extract_landmarks(self, pose_results, hands_results):
        """Extrae y normaliza los landmarks de pose y manos"""
        landmarks = []

        if pose_results.pose_landmarks:
            pose_landmarks = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
            landmarks.extend(np.array(pose_landmarks).flatten())
        else:
            landmarks.extend([0] * self.n_pose_landmarks)

        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks[:2]:
                hand_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hand_landmarks_list.extend(np.array(hand_points).flatten())

        while len(hand_landmarks_list) < self.n_hand_landmarks * 2:
            hand_landmarks_list.extend([0] * self.n_hand_landmarks)

        landmarks.extend(hand_landmarks_list)
        return np.array(landmarks)

    def collect_data(self, sign_name):
        """Recolecta secuencias de movimiento para una seña específica"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        total_sequences = int(input("Número de secuencias a recolectar: "))

        sequence_count = 0
        frame_count = 0
        is_recording = False
        current_sequence = []

        while sequence_count < total_sequences:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            pose_results, hands_results = self.process_frame(frame)

            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if is_recording:
                landmarks = self.extract_landmarks(pose_results, hands_results)
                current_sequence.append(landmarks)
                frame_count += 1

                if frame_count >= self.sequence_length:
                    sequence_data = np.array(current_sequence)
                    np.save(os.path.join(sign_dir, f"sequence_{sequence_count}.npy"), sequence_data)
                    sequence_count += 1
                    frame_count = 0
                    is_recording = False
                    current_sequence = []

            cv2.putText(frame, f"Secuencias: {sequence_count}/{total_sequences}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Recolección de Datos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and not is_recording:
                is_recording = True
                current_sequence = []
                frame_count = 0
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def evaluate(self):
        """Evalúa el modelo en tiempo real"""
        if not os.path.exists(self.model_file):
            print("No se encontró el modelo entrenado")
            return

        model = tf.keras.models.load_model(self.model_file)
        class_names = sorted(os.listdir(self.data_dir))

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        current_sequence = deque(maxlen=self.sequence_length)
        predictions_buffer = deque(maxlen=5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            pose_results, hands_results = self.process_frame(frame)

            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            landmarks = self.extract_landmarks(pose_results, hands_results)
            current_sequence.append(landmarks)

            if len(current_sequence) == self.sequence_length:
                sequence_data = np.array([current_sequence])
                prediction = model.predict(sequence_data, verbose=0)
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])
                predictions_buffer.append(predicted_class)

                most_common = max(set(predictions_buffer), key=predictions_buffer.count)
                cv2.putText(frame, f"Seña: {most_common}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Evaluación", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SignLanguageSystem()
    system.evaluate()

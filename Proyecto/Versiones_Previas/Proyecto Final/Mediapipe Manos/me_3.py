import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf


# Función para inicializar MediaPipe
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Detección de Manos", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Salir con 'ESC'
            break

    cap.release()
    cv2.destroyAllWindows()


# Función para recolectar datos
def collect_data():
    gesture = input("Ingrese el nombre del gesto que desea capturar (e.g., letra o palabra): ")
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    save_dir = os.path.join(dataset_dir, gesture)
    os.makedirs(save_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                landmarks = np.array(landmarks).flatten()
                np.save(os.path.join(save_dir, f"frame_{counter}.npy"), landmarks)
                counter += 1

        cv2.imshow("Recolección de Datos", frame)
        if cv2.waitKey(1) & 0xFF == 27 or counter >= 100:  # Salir con 'ESC' o después de 100 frames
            break

    cap.release()
    cv2.destroyAllWindows()


# Función para entrenar el modelo
def train_model():
    def load_data(dataset_dir):
        data = []
        labels = []
        gestures = []
        for label, gesture in enumerate(os.listdir(dataset_dir)):
            gesture_dir = os.path.join(dataset_dir, gesture)
            gestures.append(gesture)
            for file in os.listdir(gesture_dir):
                filepath = os.path.join(gesture_dir, file)
                landmarks = np.load(filepath)
                data.append(landmarks)
                labels.append(label)
        return np.array(data), np.array(labels), gestures

    X, y, gestures = load_data("dataset")
    y = tf.keras.utils.to_categorical(y, num_classes=len(os.listdir("dataset")))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(gestures), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    model.save("gesture_model.h5")
    print("Modelo entrenado y guardado como 'gesture_model.h5'")


# Menú principal con match-case
def main_menu():
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Inicializar MediaPipe y detectar manos")
        print("2. Recolectar datos para un gesto")
        print("3. Entrenar el modelo")
        print("4. Salir")
        choice = input("Seleccione una opción: ")

        match choice:
            case "1":
                print("Inicializando MediaPipe...")
                initialize_mediapipe()
            case "2":
                print("Recolectando datos...")
                collect_data()
            case "3":
                print("Entrenando el modelo...")
                train_model()
            case "4":
                print("Saliendo del programa. ¡Hasta luego!")
                break
            case _:
                print("Opción no válida. Intente nuevamente.")


# Ejecutar el menú principal
if __name__ == "__main__":
    main_menu()

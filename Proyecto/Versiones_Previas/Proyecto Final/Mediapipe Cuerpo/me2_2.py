import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os

# Inicializar componentes
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Variables globales
data_collection_folder = "data_2"
model_file = "sign_language_model.h5"
MAX_LANDMARKS = 1000  # Máxima longitud para los datos visibles

# Funciones auxiliares
def process_frame(image):
    """Procesa el cuadro para detectar el cuerpo y las manos con Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(image)
    hands_result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, pose_result, hands_result

def normalize_landmarks(landmarks, max_length):
    """Rellena o corta la lista de landmarks para ajustarla a un tamaño fijo."""
    if len(landmarks) > max_length:
        return landmarks[:max_length]
    return landmarks + [0] * (max_length - len(landmarks))

def collect_data(sign_name):
    """Recolecta datos de la seña especificada."""
    os.makedirs(f"{data_collection_folder}/{sign_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    total_samples = int(input("Ingrese el número total de muestras: "))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No se pudo capturar el frame.")
            break

        frame, pose_result, hands_result = process_frame(frame)

        combined_landmarks = []

        # Recolectar puntos del cuerpo
        if pose_result.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark]).flatten()
            combined_landmarks.extend(pose_landmarks)

        # Recolectar puntos de las manos
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                combined_landmarks.extend(hand_points)

        # Normalizar landmarks
        combined_landmarks = normalize_landmarks(combined_landmarks, MAX_LANDMARKS)

        if len(combined_landmarks) > 0:  # Guardar solo si hay datos
            np.savetxt(f"{data_collection_folder}/{sign_name}/{count}.txt", combined_landmarks)
            count += 1

        # Mostrar cantidad de muestras recolectadas
        cv2.putText(frame, f"Muestras: {count}/{total_samples}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow(f"Recolectando datos: {sign_name}", frame)

        if count >= total_samples:
            print(f"Recolección completada para la seña: {sign_name}")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
            print("Recolección interrumpida por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model():
    """Entrena la IA utilizando los datos recolectados."""
    X, y = [], []
    labels = {}
    max_landmarks = MAX_LANDMARKS  # Usar el valor definido globalmente

    for idx, sign_name in enumerate(os.listdir(data_collection_folder)):
        labels[idx] = sign_name
        for file in os.listdir(f"{data_collection_folder}/{sign_name}"):
            landmarks = np.loadtxt(f"{data_collection_folder}/{sign_name}/{file}")
            
            # Rellenar o cortar landmarks
            landmarks = normalize_landmarks(landmarks, max_landmarks)
            X.append(landmarks)
            y.append(idx)

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    # Definir el modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_landmarks,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar con early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    model.save(model_file)
    print("Modelo entrenado y guardado.")

def evaluate_model():
    """Evalúa el modelo en tiempo real (solo con los puntos visibles)."""
    if not os.path.exists(model_file):
        print("Primero entrene el modelo.")
        return

    model = tf.keras.models.load_model(model_file)
    labels = {idx: sign_name for idx, sign_name in enumerate(os.listdir(data_collection_folder))}

    cap = cv2.VideoCapture(0)
    last_prediction = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, pose_result, hands_result = process_frame(frame)

        combined_landmarks = []

        # Recolectar puntos del cuerpo
        if pose_result.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks.landmark]).flatten()
            combined_landmarks.extend(pose_landmarks)

        # Recolectar puntos de la mano
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                combined_landmarks.extend(hand_points)

        # Normalizar landmarks
        combined_landmarks = normalize_landmarks(combined_landmarks, MAX_LANDMARKS)

        if len(combined_landmarks) > 0:  # Solo procesar si hay puntos detectados
            combined_landmarks = np.array(combined_landmarks)
            prediction = model.predict(np.expand_dims(combined_landmarks, axis=0))
            predicted_label = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            if confidence > 0.7 and predicted_label != last_prediction:
                last_prediction = predicted_label
                engine.say(predicted_label)
                engine.runAndWait()

            cv2.putText(frame, f"{predicted_label}: {confidence*100:.2f}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Evaluando", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    """Menu principal."""
    while True:
        print("\nMenu:")
        print("1. Ver funcionalidad de Mediapipe")
        print("2. Recolección de datos")
        print("3. Entrenamiento de la IA")
        print("4. Evaluación en tiempo real")
        print("5. Salir")
        choice = input("Seleccione una opción: ")

        if choice == "1":
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame, pose_result, hands_result = process_frame(frame)

                # Dibujar cuerpo y manos
                if pose_result.pose_landmarks:
                    mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if hands_result.multi_hand_landmarks:
                    for hand_landmarks in hands_result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Mediapipe", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
                    break

            cap.release()
            cv2.destroyAllWindows()
        elif choice == "2":
            sign_name = input("Ingrese el nombre de la seña a recolectar: ")
            collect_data(sign_name)
        elif choice == "3":
            train_model()
        elif choice == "4":
            evaluate_model()
        elif choice == "5":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main_menu()

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pyttsx3
import os

# Inicializar componentes
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Variables globales
data_collection_folder = "data"
model_file = "sign_language_model.h5"

# Funciones auxiliares
def process_frame(image):
    """Procesa el cuadro para detectar manos con Mediapipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def collect_data(sign_name):
    """Recolecta datos de la seña especificada."""
    os.makedirs(f"{data_collection_folder}/{sign_name}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    total_samples = 200  # Total de muestras necesarias

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, result = process_frame(frame)
        if result.multi_hand_landmarks:
            combined_landmarks = []

            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                combined_landmarks.extend(landmarks)

            # Si solo hay una mano, rellenamos con ceros hasta completar 126 puntos
            if len(combined_landmarks) == 63:  # Solo una mano detectada
                combined_landmarks = np.concatenate([combined_landmarks, np.zeros_like(combined_landmarks)], axis=0)

            if len(combined_landmarks) == 126:  # Si tiene 126 puntos, guardamos
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
    # Preparar datos
    X, y = [], []
    labels = {}
    for idx, sign_name in enumerate(os.listdir(data_collection_folder)):
        labels[idx] = sign_name
        for file in os.listdir(f"{data_collection_folder}/{sign_name}"):
            landmarks = np.loadtxt(f"{data_collection_folder}/{sign_name}/{file}")
            X.append(landmarks)
            y.append(idx)

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    # Definir el modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(126,)),  # Soporta 126 puntos (una o dos manos)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save(model_file)
    print("Modelo entrenado y guardado.")

def evaluate_model():
    """Evalúa el modelo en tiempo real."""
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

        frame, result = process_frame(frame)
        if result.multi_hand_landmarks:
            combined_landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                combined_landmarks.extend(landmarks)

            # Si solo hay una mano, rellenamos con ceros hasta completar 126 puntos
            if len(combined_landmarks) == 63:  # Solo una mano detectada
                combined_landmarks = np.concatenate([combined_landmarks, np.zeros_like(combined_landmarks)], axis=0)

            if len(combined_landmarks) == 126:  # Procesar solo si tenemos los 126 puntos
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

                frame, result = process_frame(frame)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
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

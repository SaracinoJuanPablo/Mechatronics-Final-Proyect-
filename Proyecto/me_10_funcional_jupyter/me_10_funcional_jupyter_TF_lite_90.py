import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ================= CONFIGURACIÓN =================
DATASET_DIR = "dataset_11_90"  # Carpeta donde se guardarán los datos de cada gesto
MODEL_PATH = "gesture_model_me_10_90_pruebas.h5"
TFLITE_MODEL_PATH = "model_quantized_90_pruebas.tflite"
NORMALIZATION_PARAMS_PATH = "normalization_params_90_pruebas.npz"

# Configura la longitud de la secuencia y la cantidad de features:
SEQUENCE_LENGTH = 90   # Puedes ajustar a 30 o 90 (asegúrate de usar el mismo valor en todo el programa)
TOTAL_LANDMARKS = 126  # 2 manos * 21 puntos * 3 coordenadas = 126

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# ================= FUNCIONES AUXILIARES =================
def get_gestures():
    """
    Retorna la lista ordenada de nombres de gestos (carpetas) en DATASET_DIR.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    return sorted([d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d))])

# ================= RECOLECCIÓN DE DATOS =================
def collect_data():
    """
    Recolecta secuencias de landmarks para un gesto específico y los guarda en DATASET_DIR.
    """
    gesture = input("Ingrese el nombre de la seña a capturar: ").strip().upper()
    try:
        num_sequences = int(input("Ingrese el número de secuencias a capturar: "))
    except ValueError:
        print("Debe ingresar un número entero.")
        return

    gesture_dir = os.path.join(DATASET_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return

    sequence = []
    seq_count = 0
    print(f"\nRecolectando datos para {gesture}. Presione 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = []
            # Procesa hasta dos manos
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            # Si hay solo una mano, completa con ceros
            if len(results.multi_hand_landmarks) < 2:
                landmarks.extend([0.0] * (TOTAL_LANDMARKS - len(landmarks)))
            sequence.append(landmarks)

            # Dibuja los landmarks en el frame para visualización
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Secuencia: {seq_count}/{num_sequences}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Recolección de Datos", frame)

        # Guarda la secuencia cuando alcanza el tamaño requerido
        if len(sequence) == SEQUENCE_LENGTH:
            np.save(os.path.join(gesture_dir, f"seq_{seq_count}.npy"), sequence)
            seq_count += 1
            sequence = []
            print(f"Secuencia {seq_count}/{num_sequences} guardada.")

        # Salir con ESC o al alcanzar el número de secuencias deseado
        if cv2.waitKey(1) & 0xFF == 27 or seq_count >= num_sequences:
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= CARGA DE DATOS =================
def load_data():
    """
    Carga todas las secuencias guardadas, verificando que tengan la forma correcta.
    Retorna X (datos), y (etiquetas) y la lista de gestos.
    """
    gestures = get_gestures()
    X, y = [], []
    for idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        files = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
        print(f"Gesto '{gesture}' - secuencias encontradas: {len(files)}")
        for file in files:
            seq_path = os.path.join(gesture_dir, file)
            sequence = np.load(seq_path)
            if sequence.shape == (SEQUENCE_LENGTH, TOTAL_LANDMARKS):
                X.append(sequence)
                y.append(idx)
            else:
                print(f"Ignorando {file} con forma {sequence.shape}")
    return np.array(X), np.array(y), gestures

# ================= ENTRENAMIENTO Y CONVERSIÓN =================
def train_model():
    """
    Carga los datos, los normaliza, entrena el modelo y lo guarda.
    Luego, convierte el modelo a TFLite.
    """
    X, y, gestures = load_data()
    if len(X) == 0:
        print("No hay datos suficientes para entrenar. Recolecta datos primero.")
        return

    # One-hot encoding para las etiquetas
    y = tf.keras.utils.to_categorical(y, num_classes=len(gestures))

    # Normalización de datos
    X_mean = np.mean(X, axis=(0, 1))
    X_std = np.std(X, axis=(0, 1))
    X_norm = (X - X_mean) / (X_std + 1e-7)
    np.savez(NORMALIZATION_PARAMS_PATH, mean=X_mean, std=X_std)
    print("Parámetros de normalización guardados.")

    # Arquitectura del modelo
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, TOTAL_LANDMARKS))
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(gestures), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("Iniciando entrenamiento...")
    history = model.fit(X_norm, y, epochs=50, batch_size=32, validation_split=0.2)

    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    # Conversión a TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Modelo TFLite exportado en {TFLITE_MODEL_PATH}")

# ================= EVALUACIÓN EN TIEMPO REAL =================
def evaluate():
    """
    Evalúa el modelo TFLite en tiempo real utilizando la cámara.
    Se muestra en pantalla la seña detectada y la confianza.
    """
    if not os.path.exists(TFLITE_MODEL_PATH):
        print("Primero entrena y exporta el modelo.")
        return

    # Cargar parámetros de normalización
    try:
        data = np.load(NORMALIZATION_PARAMS_PATH)
        X_mean, X_std = data["mean"], data["std"]
    except Exception as e:
        print(f"Error al cargar la normalización: {e}")
        return

    # Asegura que los gestos estén ordenados
    gestures = get_gestures()
    print("Gestos para evaluación:", gestures)

    # Configurar el intérprete TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede acceder a la cámara.")
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    current_gesture = None
    current_confidence = 0.0

    print("Iniciando evaluación. Presiona 'ESC' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) < 2:
                landmarks.extend([0.0] * (TOTAL_LANDMARKS - len(landmarks)))
            sequence.append(landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Realiza la predicción cada ciertos frames cuando se tiene una secuencia completa
        if len(sequence) == SEQUENCE_LENGTH and frame_count % 10 == 0:
            seq_array = np.array(sequence)
            seq_norm = (seq_array - X_mean) / (X_std + 1e-7)
            input_data = seq_norm.reshape(1, SEQUENCE_LENGTH, TOTAL_LANDMARKS).astype(np.float32)

            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details['index'])[0]
            pred_idx = np.argmax(prediction)
            confidence = prediction[pred_idx]

            if confidence > 0.65:
                current_gesture = gestures[pred_idx]
                current_confidence = confidence
            else:
                current_gesture = None

        # Mostrar el resultado en el frame
        if current_gesture:
            cv2.putText(frame, f"{current_gesture} ({current_confidence:.2%})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Evaluación", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Evaluación terminada.")

# ================= MENÚ PRINCIPAL =================
def menu():
    while True:
        print("\n===== MENÚ PRINCIPAL =====")
        print("1. Recolectar datos")
        print("2. Entrenar modelo y exportar a TFLite")
        print("3. Evaluar en tiempo real")
        print("4. Salir")
        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            collect_data()
        elif opcion == "2":
            train_model()
        elif opcion == "3":
            evaluate()
        elif opcion == "4":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intente de nuevo.")

# ================= PROGRAMA PRINCIPAL =================
if __name__ == "__main__":
    menu()

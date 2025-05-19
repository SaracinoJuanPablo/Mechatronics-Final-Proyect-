# ------------- PROYECTO FINAL G & S--------------------
## LIBRERIAS
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Comunicación y cámara
import socket
import queue

# Voz
import pyttsx3
import threading

# Módulos para reconocimiento de voz
import speech_recognition as sr
import librosa
import io
import wave

# Generador de audios
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
## UDP
# Configuración para el reconocimiento de voz
SAMPLE_RATE_IN = 48000  # Tasa del micrófono INMP441
SAMPLE_RATE_OUT = 16000  # Tasa requerida por la API de reconocimiento
BUFFER_DURATION = 5  # segundos


UDP_IP_PI = "192.168.7.2"  # IP de la Raspberry Pi
UDP_OPEN = '0.0.0.0'

# Puertos para diferentes servicios
UDP_PORT_MICROFONO = 5006
UDP_PORT_TEXT = 5005
UDP_PORT_SERVO = 5001  # Puerto para enviar comandos
UDP_PORT_PARLANTE = 5003
UDP_PORT_CAM = 5002  # Puerto para recibir video
MAX_PACKET_SIZE = 1400  # Tamaño máximo del paquete UDP


## CAMARA
### MEDIAPIPE
# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5, #probar con 0.4
    min_tracking_confidence=0.5 #probar con 0.4
)
# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
### COMUNICACION CAMARA
'''class UDPCamera:
    def __init__(self):
        self.host = UDP_OPEN
        self.port = UDP_PORT_CAM
        self.buffer_size = 65536
        self.mtu = 1400
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2)
        self.frame = None
        self.fragments = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Configuración INTRÍNSECA de MediaPipe para el seguimiento
        self.hands_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Socket para enviar datos del servo
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.sock.bind((self.host, self.port))
            self.thread = threading.Thread(target=self._receive_frames, daemon=True)
            self.thread.start()

    def _process_hand(self, frame):
        # Procesamiento específico de la muñeca
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_tracker.process(frame_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_handedness in results.multi_handedness:
                if hand_handedness.classification[0].label == 'Left':
                    wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
                    # Mapear coordenadas de la palma a un rango de -7.5 a 7.5
                    x_normalized = int((wrist.x - 0.5) * 15) 
                    
                    # Envío UDP automático
                    self.send_sock.sendto(
                        str(x_normalized).encode(), 
                        (UDP_IP_PI, UDP_PORT_SERVO)
                    )
                    
                    # Dibujar punto (opcional)
                    wrist_pixel = mp_drawing._normalized_to_pixel_coordinates(
                        wrist.x, wrist.y, frame.shape[1], frame.shape[0]
                    )
                    if wrist_pixel:
                        cv2.circle(frame, wrist_pixel, 10, (0, 255, 0), -1)
                    
                    return x_normalized
        return None

    def _receive_frames(self):
        while self.running:
            try:
                fragment, _ = self.sock.recvfrom(self.buffer_size)
                with self.lock:
                    self.fragments.append(fragment)
                    if len(fragment) < self.mtu:
                        frame_bytes = b''.join(self.fragments)
                        self.fragments = []
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        # Procesamiento AUTOMÁTICO de la mano
                        if frame is not None:
                            self._process_hand(frame)
                            self.frame = frame  # Almacenar frame procesado
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error: {str(e)}")
                break

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        self.hands_tracker.close()
        with self.lock:
            self.fragments = []
            self.frame = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.sock.close()
        self.send_sock.close()

    def __del__(self):
        self.release()'''
class UDPCamera:
    def __init__(self):
        self.host = UDP_OPEN
        self.port = UDP_PORT_CAM
        self.buffer_size = 65536
        self.mtu = 1400
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2)
        self.frame = None
        self.fragments = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.sock.bind((self.host, self.port))
            self.thread = threading.Thread(target=self._receive_frames, daemon=True)
            self.thread.start()

    def _receive_frames(self):
        while self.running:
            try:
                fragment, _ = self.sock.recvfrom(self.buffer_size)
                with self.lock:
                    self.fragments.append(fragment)
                    if len(fragment) < self.mtu:  # Último fragmento
                        frame_bytes = b''.join(self.fragments)
                        self.fragments = []
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        self.frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error en recepción: {str(e)}")
                break

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        with self.lock:
            self.fragments = []
            self.frame = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.sock.close()

    def __del__(self):
        self.release()
### MODELO TFLITE
class TFLiteModel:
    def __init__(self, model_path):
        # Cargar el modelo TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obtener detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, input_data):
        # Asegurar el tipo de dato correcto y agregar dimensión batch si es necesario
        input_data = np.array(input_data, dtype=self.input_details[0]['dtype'])
        if len(input_data.shape) == len(self.input_details[0]['shape']) - 1:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Establecer la entrada y ejecutar la inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Obtener la salida
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

# Configuración de TensorFlow para rendimiento
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    # Configuración de TensorFlow para rendimiento en CPU
    try:
        # Verificar si hay GPU disponible (para futuras expansiones)
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            # Configuración para GPU (no se ejecutará en tu caso)
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU disponible para aceleración")
        else:
            # Optimización para CPU
            tf.config.threading.set_intra_op_parallelism_threads(4)  # Aprovecha núcleos físicos
            tf.config.threading.set_inter_op_parallelism_threads(2)  # Paralelismo entre operaciones
            print("Modo CPU activado: Configuración optimizada para Intel Core i7-7500U")
            
    except Exception as e:
        print(f"Error de configuración: {str(e)}")
        print("Usando configuración por defecto de CPU")
### ARCHIVOS
# Configuración de directorios y archivos
data_dir = "hand_gestures_data_v15"
os.makedirs(data_dir, exist_ok=True)

# Modelo y datos de entrenamiento
model = None
# Inicializar scaler y label encoder
scaler = StandardScaler()
label_encoder = LabelEncoder()
model_file = "hand_gesture_nn_model_v15_POSE.h5"
scaler_file = "hand_gesture_scaler_v15_POSE.pkl"
encoder_file = "hand_gesture_encoder_v15_POSE.pkl"
gesture_data = "gesture_data_v15_POSE.pkl" 
model_tflite = "modelo_optimizadotl_v15_POSE.tflite"

# Variables globales para estado
data = []
labels = []

# Estado del sistema
is_trained = False
is_collecting = False
current_gesture = ""
samples_collected = 0
max_samples = 1000

# Control de tiempo para la recolección continua
last_sample_time = 0
sample_delay = 0.05  # 50ms entre muestras

# Temporizador para mostrar mensajes
message = ""
message_until = 0

# Para evaluación del modelo
metrics = {
    'accuracy': 0,
    'val_accuracy': 0,
    'training_time': 0
}
### EXTRACCION DE LANDMARKS
'''def extract_hand_landmarks(frame):
    """
    Extrae los landmarks de las manos desde un frame de video.

    Args:
        frame: Imagen capturada por la cámara (en formato BGR).

    Returns:
        tuple: Lista de landmarks normalizados (126 elementos) y booleano indicando si se detectaron manos.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    landmarks_data = []
    hands_detected = False
    
    if results.multi_hand_landmarks:
        hands_detected = True
        # Extraer landmarks de hasta dos manos
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks en el frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []

            # Extraer coordenadas (x,y,z) de los 21 landmarks
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            landmarks_data.extend(landmarks)
    
    # Normalizar para 2 manos (si solo hay una o ninguna, rellenar con ceros)
    while len(landmarks_data) < 21 * 3 * 2:  # 21 landmarks * 3 coordenadas * 2 manos
        landmarks_data.append(0.0)
    
    # Limitar a exactamente 126 valores (21 landmarks * 3 coordenadas * 2 manos)
    landmarks_data = landmarks_data[:21 * 3 * 2]
    
    return landmarks_data, hands_detected'''

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # socket de los puntos de la muñeca

def extract_hand_pose_landmarks(frame, send_sock=None):
    """
    Extrae landmarks de las manos y del cuerpo, y realiza seguimiento de la muñeca derecha.
    
    Args:
        frame: Imagen en formato BGR
        send_sock: Socket UDP opcional para enviar datos de seguimiento
    
    Returns:
        tuple: (landmarks_data, hands_detected, pose_detected)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar manos
    hands_results = hands.process(frame_rgb)
    
    # Procesar pose
    pose_results = pose.process(frame_rgb)
    
    # Inicializar variables
    hand_landmarks_data = []
    pose_landmarks_data = []
    hands_detected = False
    pose_detected = False
    x_normalized = None
    right_wrist_pixel = None
    
    # Extraer landmarks de manos
    if hands_results.multi_hand_landmarks:
        hands_detected = True
        # Procesar ambas manos para landmarks
        for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            # Dibujar landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extraer coordenadas
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            hand_landmarks_data.extend(landmarks)
            
            # Detectar mano derecha para seguimiento
            if hands_results.multi_handedness:
                handedness = hands_results.multi_handedness[hand_idx]
                if handedness.classification[0].label == 'Left' and not x_normalized:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    
                    # Calcular coordenadas normalizadas
                    x_normalized = int((wrist.x - 0.5) * 15)  # Rango -7.5 a 7.5
                    print(x_normalized)
                    
                    # Obtener coordenadas para dibujo
                    right_wrist_pixel = mp_drawing._normalized_to_pixel_coordinates(
                        wrist.x, wrist.y, frame.shape[1], frame.shape[0]
                    )

        # Enviar datos y dibujar si se detectó mano derecha
        if x_normalized is not None:
            if send_sock:  # Solo enviar si se provee socket
                send_sock.sendto(
                    str(x_normalized).encode(),
                    (UDP_IP_PI, UDP_PORT_SERVO)  # Asegurar que estas constantes están definidas
                )
            if right_wrist_pixel:
                cv2.circle(frame, right_wrist_pixel, 10, (0, 255, 0), -1)
    
    # Rellenar con ceros si no hay manos
    while len(hand_landmarks_data) < 21 * 3 * 2:  # 21 landmarks * 3 coordenadas * 2 manos
        hand_landmarks_data.append(0.0)
    hand_landmarks_data = hand_landmarks_data[:21 * 3 * 2]  # Limitar a exactamente 126 valores
    
    # Extraer landmarks de pose
    if pose_results.pose_landmarks:
        pose_detected = True
        # Dibujar landmarks de pose
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
        
        # Extraer coordenadas de los 33 landmarks de pose
        for landmark in pose_results.pose_landmarks.landmark:
            pose_landmarks_data.extend([landmark.x, landmark.y, landmark.z])
    
    # Rellenar con ceros si no hay pose detectada
    while len(pose_landmarks_data) < 33 * 3:  # 33 landmarks * 3 coordenadas
        pose_landmarks_data.append(0.0)
    pose_landmarks_data = pose_landmarks_data[:33 * 3]  # Limitar a exactamente 99 valores
    
    # Concatenar landmarks de manos y pose
    combined_landmarks = hand_landmarks_data + pose_landmarks_data
    
    return combined_landmarks, hands_detected, pose_detected
def set_message(message_text, duration=2):
    global message, message_until
    message = message_text
    message_until = time.time() + duration
### RECOLECCION
def start_collection(gesture_name):
    global is_collecting, current_gesture, samples_collected
    is_collecting = True
    current_gesture = gesture_name
    samples_collected = 0
    set_message(f"Mantenga la seña frente a la cámara. Recolectando '{gesture_name}'...", 3)

def stop_collection():
    global is_collecting, current_gesture, samples_collected
    is_collecting = False
    current_gesture = ""
    samples_collected = 0
    set_message("Recolección finalizada", 2)
### GUARDADO DE DATOS
def save_data():
    global data, labels
    data_to_save = {"features": data, "labels": labels}
    with open(f"{data_dir}/{gesture_data}", "wb") as f:
        pickle.dump(data_to_save, f)
    set_message(f"Datos guardados: {len(data)} muestras", 1)
### RECOLECCION DE MUESTRAS
def collect_sample(landmarks):
    global is_collecting, samples_collected, last_sample_time, data, labels
    
    if not is_collecting:
        return False
    
    current_time = time.time()
    if current_time - last_sample_time >= sample_delay:
        data.append(landmarks)
        labels.append(current_gesture)
        samples_collected += 1
        last_sample_time = current_time
        
        if samples_collected % 10 == 0:
            save_data()
        
        if samples_collected >= max_samples:
            stop_collection()
            return True
    
    return False
### CARGA DE DATOS
def load_data():
    global data, labels
    try:
        with open(f"{data_dir}/{gesture_data}", "rb") as f:
            loaded_data = pickle.load(f)
            data = loaded_data["features"]
            labels = loaded_data["labels"]
        set_message(f"Datos cargados: {len(data)} muestras", 2)
        return True
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        set_message("No se encontraron datos previos", 2)
        return False
### RED NEURONAL
def check_model_exists():
    return os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(encoder_file)

def create_neural_network(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
### ENTRENAMIENTO
def train_model():
    global model, scaler, label_encoder, metrics, is_trained
    
    if len(data) < 10:
        set_message("Se necesitan más datos para entrenar", 2)
        return False
    
    X = np.array(data)
    y = np.array(labels)
    
    # Codificar etiquetas
    y_encoded = label_encoder.fit_transform(y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalizar datos
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Crear y entrenar modelo
    num_classes = len(set(y_encoded))
    set_message(f"Entrenando modelo con {num_classes} clases...", 2)
    
    model = create_neural_network(X_train.shape[1], num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluar modelo
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar métricas
    metrics['accuracy'] = accuracy
    metrics['val_accuracy'] = max(history.history['val_accuracy'])
    metrics['training_time'] = training_time
    
    # Guardar modelo y preprocesadores
    model.save(model_file)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    set_message(f"Modelo entrenado con precisión: {accuracy:.2%}", 3)
    is_trained = True
    
    return True

### CARGA DEL MODELO ENTRENADO
def load_saved_model():
    global scaler, label_encoder
    try:
        model = load_model(model_file)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
        set_message("Modelo cargado correctamente", 2)
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        set_message("Error al cargar el modelo", 2)
        return None

### MODELO DE TFLITE
def predict_tflite(landmarks, tflite_model, scaler, label_encoder, threshold=0.5):
    try:
        # Preprocesar los landmarks
        landmarks_array = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = scaler.transform(landmarks_array)
        
        # Realizar predicción
        predictions = tflite_model.predict(landmarks_scaled)[0]
        
        # Obtener la clase con mayor probabilidad
        max_prob_idx = np.argmax(predictions)
        confidence = predictions[max_prob_idx]
        
        if confidence >= threshold:
            # Decodificar la etiqueta
            predicted_label = label_encoder.inverse_transform([max_prob_idx])[0]
            return predicted_label, confidence
        else:
            return "Desconocido", confidence
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return "Error", 0.0

### CONVERSION A TFLITE
def convert_to_tflite(model_file, model_tflite):
    try:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"El archivo {model_file} no existe.")
        
        # Cargar el modelo entrenado
        modelo = load_model(model_file)
        
        # Convertir a TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
        tflite_model = converter.convert()
        
        # Guardar el modelo convertido
        with open(model_tflite, "wb") as f:
            f.write(tflite_model)
        
        print("Modelo convertido a TensorFlow Lite.")
    except Exception as e:
        print("Error al convertir el modelo a TFLite:", e)
### ELIMINADOR DE SEÑAS
def delete_gesture(target_label):
    global data, labels
    if target_label not in labels:
        print(f"Error: La etiqueta '{target_label}' no existe")
        return False
    
    # Filtrar elementos a mantener
    new_data = []
    new_labels = []
    deleted_count = 0
    
    for feature, label in zip(data, labels):
        if label == target_label:
            deleted_count += 1
        else:
            new_data.append(feature)
            new_labels.append(label)
    
    # Actualizar listas globales
    data.clear()
    labels.clear()
    data.extend(new_data)
    labels.extend(new_labels)
    
    print(f"Se eliminaron {deleted_count} muestras de '{target_label}'")
    return True
### GENERADOR DE AUDIOS
# Configuración de directorios y archivos
audio_dir = "pyttsx3_audios"
os.makedirs(audio_dir, exist_ok=True)

# Configurar motor TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def trim_audio_silence(file_path):
    """Recorta silencios al inicio y final del audio"""
    audio = AudioSegment.from_file(file_path, format="wav")

    # Parámetros ajustables
    config = {
        'min_silence_len': 200,     # 200 ms de silencio mínimo para considerar corte
        'silence_thresh': -45,      # -45 dB de umbral de silencio
        'end_buffer': 150           # 150 ms extra al final
    }
    
    # Detectar segmentos no silenciosos
    nonsilent_parts = detect_nonsilent(
        audio,
        min_silence_len=config['min_silence_len'], # Duración mínima de silencio a considerar (ms)
        silence_thresh=config['silence_thresh'] # Umbral de volumen para considerar silencio (dB)
    )
    
    if nonsilent_parts:
        start = max(0, nonsilent_parts[0][0] - 50)  # 50 ms buffer inicial
        end = nonsilent_parts[-1][1] + config['end_buffer']
        trimmed_audio = audio[start:end]
        trimmed_audio.export(file_path, format="wav")

def compilador_audios(label):
    """Genera y ajusta audio para eliminar silencios"""
    nombre_archivo = label.replace(' ', '_').lower() + '.wav'
    ruta_audio = os.path.join(audio_dir, nombre_archivo)
    
    if os.path.exists(ruta_audio):
        return
    
    temp_path = os.path.join(audio_dir, "temp.wav")
    try:
        # Generar audio temporal
        engine.save_to_file(label, temp_path)
        engine.runAndWait()
        
        # Recortar y renombrar
        trim_audio_silence(temp_path)
        os.rename(temp_path, ruta_audio)
        print(f"Audio generado: {nombre_archivo}")
        
    except Exception as e:
        print(f"Error generando {label}: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generar_audios():
    """Genera audios para todas las etiquetas únicas"""
    etiquetas_unicas = set(labels)
    print("\nGenerando audios para señas...")
    
    for label in etiquetas_unicas:
        compilador_audios(label)
    
    print("Proceso de generación de audios completado\n")

### MENU
def print_menu():
    print("\n=== MENU PRINCIPAL ===")
    print("1. Recolectar nueva seña")
    print("2. Entrenar modelo")
    print("3. Listar señas cargadas")
    print("4. Eliminar señas")
    print("5. Generar Audios")
    print("0. Salir")

### LISTADO DE GESTOS
def list_gestures():
    # Asumiendo que 'labels' es la lista donde se guardan las señas
    if not labels:
        print("No hay señas guardadas.")
    else:
        unique_gestures = list(set(labels))
        print("\n--- Señas Guardadas ---")
        for i, gesture in enumerate(unique_gestures, 1):
            print(f"{i}. {gesture}")

### RECOLECCION DE SEÑAS
def run_collection_mode():
    # Inicia la cámara
    try:
        cap = UDPCamera()
        print("Cámara UDP iniciada para recolección.")
    except Exception as e:
        print(f"Error al iniciar la cámara: {str(e)}")
        return
    
    while is_collecting:  # Asumiendo que 'is_collecting' se activa en start_collection()
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.1)
            continue

        # Procesar frame actual
        landmarks, hands_detected, pose_detected = extract_hand_pose_landmarks(frame, send_sock)
        
        frame_h, frame_w, _ = frame.shape
        
        # Mostrar información en pantalla durante la recolección
        progress = int((samples_collected / max_samples) * frame_w)
        cv2.rectangle(frame, (0, 0), (progress, 20), (0, 255, 0), -1)
        cv2.putText(frame, f"Recolectando: {current_gesture} ({samples_collected}/{max_samples})", 
                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar información de detección
        detection_info = []
        if hands_detected:
            detection_info.append("Manos")
        if pose_detected:
            detection_info.append("Pose")
            
        detection_text = ", ".join(detection_info) if detection_info else "Nada detectado"
        cv2.putText(frame, f"Detectado: {detection_text}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if hands_detected:
            collect_sample(landmarks)
        else:
            cv2.putText(frame, "¡Muestra las manos!", (frame_w//2 - 100, frame_h//2), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if not is_collecting:  # Cuando termina la recolección
            menu_active = True
            save_data()
        
        cv2.imshow("Recolectar Señas", frame)
        
        key = cv2.waitKey(1)
        # Puedes agregar una tecla para finalizar la recolección, por ejemplo 'm' para volver al menú.
        if key == ord('m'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
### FUNCION PRINCIPAL 
def main():
    global model, is_trained, data, labels
    
    # Inicialización del sistema
    is_trained = False
    model = None
    data = []
    labels = []
    
    # Cargar datos existentes
    load_data()
    
    # Intentar cargar modelo si existe
    if check_model_exists():
        model = load_saved_model()
        is_trained = True
    else:
        is_trained = False

    # Mostrar el menú en la consola
    print_menu()

    # Bucle principal de selección en consola
    while True:
        opcion = input("\nSelecciona una opción (Recolectar: 1, Entrenar: 2, Listar: 3, Eliminar: 4, Gen.Audio: 5, Salir: 0): ").strip()
        
        if opcion == '1':
            # Recolección de señas
            gesture_name = input("Ingrese nombre de la seña (ej. 'Hola'): ")
            if gesture_name:
                start_collection(gesture_name)
                # Iniciar la cámara para mostrar video durante la recolección
                run_collection_mode()
                
        elif opcion == '2':
            if len(data) > 10:
                train_model()
                model = load_saved_model() if check_model_exists() else None
                is_trained = True
                print("Entrenamiento completado. Modelo entrenado.")
                convert_to_tflite(model_file, model_tflite)
                print("Convertido a TFLite para evaluación en tiempo real")
            else:
                print("¡Necesitas al menos 10 muestras para entrenar!")
                
        elif opcion == '3':
            list_gestures()  # Lista las señas cargadas

        elif opcion == '4':
            if not labels:
                print("No hay señas guardadas para eliminar")
                exit()
            print("\n--- Señas Registradas ---")
            for label in set(labels):
                print(f"- {label}")
            target_label = input("\nIngrese el nombre exacto de la seña a eliminar: ").strip()
            if delete_gesture(target_label):
                save_data()
            else:
                print("No se realizaron cambios en los datos")
        
        elif opcion == '5':
            if not labels:
                print("No hay señas guardadas")
                exit()
            # Generar audios automáticamente
            generar_audios()
            

                
        elif opcion == '0':
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida, intenta nuevamente.")
        
        # Mostrar nuevamente el menú luego de finalizar la opción seleccionada.
        print_menu()
if __name__ == "__main__":
    main()
    save_data()  # Guarda los datos recolectados
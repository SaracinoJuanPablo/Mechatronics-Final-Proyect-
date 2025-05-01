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


## VOZ A PANTALLA
### RECONOCEDOR DE VOZ
# Inicializar el reconocedor de voz
recognizer = sr.Recognizer()

# Configuración UDP para voz
sock_voice = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
microfono_queue = queue.Queue()

# Variable para controlar el servicio de reconocimiento de voz
speech_recognition_running = True #VA EN EL WHILE.

# Variable para almacenar la última transcripción
last_transcription = "" #NO ESTAN EN SPEECH-TO-TEXT-FREE-UDP-V1.PY

def recibir_audio():
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_audio.bind(("0.0.0.0", UDP_PORT_MICROFONO))
        
        buffer = bytearray()
        bytes_needed = SAMPLE_RATE_IN * 4 * BUFFER_DURATION  # 4 bytes por muestra (32-bit)
        
        while speech_recognition_running:
            data, _ = sock_audio.recvfrom(4096)
            buffer.extend(data)
            
            while len(buffer) >= bytes_needed:
                # Extraer 5 segundos de audio
                chunk = bytes(buffer[:bytes_needed])
                del buffer[:bytes_needed]
                
                # Convertir a formato numpy
                audio_int32 = np.frombuffer(chunk, dtype=np.int32)
                audio_float32 = audio_int32.astype(np.float32) / 2**31
                
                # Remuestrear a 16kHz
                audio_16k = librosa.resample(
                    audio_float32,
                    orig_sr=SAMPLE_RATE_IN,
                    target_sr=SAMPLE_RATE_OUT
                )
                
                # Convertir a int16 para la API de reconocimiento
                audio_int16 = (audio_16k * 32767).astype(np.int16)
                
                # Crear un archivo WAV en memoria
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes por muestra (16 bits)
                    wav_file.setframerate(SAMPLE_RATE_OUT)
                    wav_file.writeframes(audio_int16.tobytes())
                
                wav_buffer.seek(0)  # Rebobinar el buffer
                microfono_queue.put(wav_buffer)
    except Exception as e:
        print(f"Error en recibir_audio: {e}")
    #NO ESTAN EN SPEECH-TO-TEXT-FREE-UDP-V1.PY  ver de comentarlo
    finally: 
        sock_audio.close() 
### PROCESAR AUDIO
def procesar_audio():
    global last_transcription #NO ESTAN EN SPEECH-TO-TEXT-FREE-UDP-V1.PY
    while speech_recognition_running:
        try:
            wav_buffer = microfono_queue.get(timeout=1)
            
            # Crear un objeto AudioData desde el buffer WAV
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)
            
            # Realizar la transcripción usando la API gratuita de Google
            transcription = recognizer.recognize_google(audio_data, language="es-ES")
            
            print(f"Transcripción: {transcription}")
            last_transcription = transcription #NO ESTAN EN SPEECH-TO-TEXT-FREE-UDP-V1.PY
            
            # Enviar transcripción por UDP si es necesario
            sock_voice.sendto(transcription.encode(), (UDP_IP_PI, UDP_PORT_TEXT))
            
        except queue.Empty:
            continue
        except sr.UnknownValueError:
            print("No se detectó voz en el audio")
        except sr.RequestError as e:
            print(f"Error en la solicitud a la API de Google: {e}")
        except Exception as e:
            print(f"Error en la transcripción: {e}")
### ACTIVACION DE RECONOCIMIENTO DE VOZ
# Función para iniciar el servicio de reconocimiento de voz
def start_speech_recognition():
    global speech_recognition_running
    speech_recognition_running = True
    
    # Iniciar hilos para el reconocimiento de voz
    audio_thread = threading.Thread(target=recibir_audio, daemon=True)
    process_thread = threading.Thread(target=procesar_audio, daemon=True)
    
    audio_thread.start()
    process_thread.start()
    
    print("Servicio de reconocimiento de voz iniciado...")
    return audio_thread, process_thread
### DETENCION DE RECONOCIMIENTO DE VOZ
# Función para detener el servicio de reconocimiento de voz
def stop_speech_recognition():
    global speech_recognition_running
    speech_recognition_running = False
    sock_voice.close()
    print("Servicio de reconocimiento de voz detenido.")
## CAMARA
### MOTOR TEXTO-VOZ
# Inicializa el socket UDP (compartido para todos los hilos)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Inicializa el motor TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Objeto de bloqueo para sincronización
tts_lock = threading.Lock()

# Variable global para última reproducción
last_spoken_gesture = None

def speak_text(text):
    global last_spoken_gesture, udp_socket

    if not hasattr(speak_text, 'queue'):
        speak_text.queue = []
        speak_text.processing = False
        speak_text.thread = None
        speak_text.event = threading.Event()  # Nuevo evento para sincronización
        speak_text.current_audio = None  # Bandera de audio en transmisión

    # Evitar duplicados y agregar a cola
    if text != last_spoken_gesture and text not in speak_text.queue:
        speak_text.queue.append(text)
        last_spoken_gesture = text

    def _process_queue():
        while speak_text.queue or speak_text.current_audio:
            # Esperar si hay audio en curso
            if speak_text.current_audio:
                time.sleep(0.1)
                continue

            # Bloquear mientras se procesa
            with tts_lock:
                if speak_text.queue:
                    speak_text.current_audio = speak_text.queue.pop(0)
                    
                    try:
                        # Generar audio
                        temp_file = "temp_audio.wav"
                        tts_engine.save_to_file(speak_text.current_audio, temp_file)
                        tts_engine.runAndWait()

                        # Calcular tiempo de audio aproximado
                        audio_duration = len(open(temp_file, 'rb').read()) / (16000 * 2)  # 16KHz, 16bits
                        
                        # Enviar por UDP
                        with open(temp_file, 'rb') as f:
                            audio_data = f.read()
                            total_chunks = (len(audio_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                            for i in range(total_chunks):
                                chunk = audio_data[i*MAX_PACKET_SIZE:(i+1)*MAX_PACKET_SIZE]
                                udp_socket.sendto(chunk, (UDP_IP_PI, UDP_PORT_PARLANTE))
                                time.sleep(0.001)

                        print(f"Audio enviado: {speak_text.current_audio}")
                        
                        # Esperar tiempo estimado de reproducción
                        time.sleep(audio_duration * 0.8)  # Margen de seguridad

                    except Exception as e:
                        print(f"Error: {str(e)}")
                    finally:
                        speak_text.current_audio = None
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

        speak_text.processing = False

    if not speak_text.processing:
        speak_text.processing = True
        speak_text.thread = threading.Thread(target=_process_queue, daemon=True)
        speak_text.thread.start()
### MEDIAPIPE
# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5, #probar con 0.4
    min_tracking_confidence=0.5 #probar con 0.4
)
mp_drawing = mp.solutions.drawing_utils
### COMUNICACION CAMARA
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
model_file = "hand_gesture_nn_model_v15.h5"
scaler_file = "hand_gesture_scaler_v15.pkl"
encoder_file = "hand_gesture_encoder_v15.pkl"
gesture_data = "gesture_data_v15.pkl" 
model_tflite = "modelo_optimizadotl_v15.tflite"

# Variables globales para estado
data = []
labels = []

# Estado del sistema
is_trained = False
is_collecting = False
current_gesture = ""
samples_collected = 0
max_samples = 5000

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
def extract_hand_landmarks(frame):
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
    
    return landmarks_data, hands_detected

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
### MENU
def print_menu():
    print("\n=== MENU PRINCIPAL ===")
    print("1. Recolectar nueva seña")
    print("2. Entrenar modelo")
    print("3. Listar señas cargadas")
    print("4. Evaluar en tiempo real")
    print("5. Salir")

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

        landmarks, hands_detected = extract_hand_landmarks(frame)

        frame_h, frame_w, _ = frame.shape

        # Mostrar información en pantalla durante la recolección
        progress = int((samples_collected / max_samples) * frame_w)
        cv2.rectangle(frame, (0, 0), (progress, 20), (0, 255, 0), -1)
        cv2.putText(frame, f"Recolectando: {current_gesture} ({samples_collected}/{max_samples})", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
### EVALUACION EN TIEMPO REAL
def run_evaluation_mode():
    global model_tflite
    # Inicializa el modelo TFLite si aún no se ha cargado
    if os.path.exists(model_tflite):
        tflite_model = TFLiteModel(model_tflite)
    else:
        print("El modelo TFLite no existe. Conviértelo primero.")
        return

    # Inicia la cámara
    try:
        cap = UDPCamera()
        print("Cámara UDP iniciada para evaluación en tiempo real.")
    except Exception as e:
        print(f"Error al iniciar la cámara: {str(e)}")
        return
    
    # Variables para el sistema de confirmación de señas
    consecutive_frames = 0
    last_prediction = ""
    confirmation_threshold = 5  # Número de frames consecutivos necesarios
    confirmed_gesture = ""
    
    # Para el enfoque alternativo de ventana deslizante
    window_size = 5
    prediction_window = []
    
    # Umbral de confianza para considerar una detección válida
    confidence_threshold = 0.9
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        landmarks, hands_detected = extract_hand_landmarks(frame)
        frame_h, frame_w, _ = frame.shape

        if hands_detected:
            prediction, confidence = predict_tflite(landmarks, tflite_model, scaler, label_encoder, threshold=confidence_threshold)
            
            # Extraer valor escalar en caso de que 'confidence' sea un array
            confidence_value = np.max(confidence) if isinstance(confidence, np.ndarray) else confidence
            
            # Color basado en la confianza
            color = (0, 255, 0) if confidence_value > confidence_threshold else (0, 165, 255)
            
            # Mostrar la predicción actual
            cv2.putText(frame, f"Seña detectada: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confianza: {confidence_value:.2%}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Sistema de confirmación por frames consecutivos
            if confidence_value > confidence_threshold and prediction != "Desconocido":
                # Enfoque 1: Frames consecutivos
                if prediction == last_prediction:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 1  # Reiniciar contador si cambia la predicción
                
                last_prediction = prediction
                
                # Actualizar la ventana deslizante para el enfoque alternativo
                prediction_window.append(prediction)
                if len(prediction_window) > window_size:
                    prediction_window.pop(0)  # Mantener solo los últimos 'window_size' elementos
                
                # Mostrar barra de progreso para la confirmación
                progress_width = int((consecutive_frames / confirmation_threshold) * 200)  # Ancho máximo de 200 píxeles
                progress_width = min(progress_width, 200)  # Limitar al máximo
                
                # Dibujar barra de progreso
                cv2.rectangle(frame, (10, 120), (210, 140), (100, 100, 100), -1)  # Fondo gris
                cv2.rectangle(frame, (10, 120), (10 + progress_width, 140), (0, 255, 0), -1)  # Barra verde
                cv2.putText(frame, f"Confirmando: {consecutive_frames}/{confirmation_threshold}", (10, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Verificar si se ha alcanzado el umbral de confirmación
                if consecutive_frames >= confirmation_threshold:
                    if confirmed_gesture != prediction:  # Evitar repeticiones
                        confirmed_gesture = prediction
                        threading.Thread(target=speak_text, args=(prediction,), daemon=True).start()
                        cv2.putText(frame, f"¡CONFIRMADO!: {prediction}", (frame_w//4, frame_h//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Enfoque alternativo: Seña más frecuente en la ventana
                if len(prediction_window) == window_size:
                    from collections import Counter
                    most_common = Counter(prediction_window).most_common(1)[0]  # (elemento, frecuencia)
                    most_common_gesture, frequency = most_common
                    
                    # Mostrar información sobre el enfoque alternativo
                    cv2.putText(frame, f"Más frecuente: {most_common_gesture} ({frequency}/{window_size})", 
                                (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                    
                    # Este enfoque se puede habilitar si se prefiere sobre el de frames consecutivos
                    # Si se quiere usar este enfoque en lugar del de frames consecutivos, descomentar:
                    # if frequency >= 3 and confirmed_gesture != most_common_gesture:  # Mayoría en ventana de 5
                    #     confirmed_gesture = most_common_gesture
                    #     threading.Thread(target=speak_text, args=(most_common_gesture,), daemon=True).start()
            else:
                # Reiniciar contador si la confianza es baja
                consecutive_frames = 0
                cv2.putText(frame, "Confianza insuficiente", (10, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            # Reiniciar contador si no se detectan manos
            consecutive_frames = 0
            cv2.putText(frame, "Acerca las manos a la cámara", (frame_w//4, frame_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(frame, "Presiona ESC para volver al menú", (10, frame_h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Evaluación en Tiempo Real", frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

### FUNCION PRINCIPAL 
def main():
    global model, is_trained, data, labels

    # Iniciar el servicio de reconocimiento de voz al iniciar el programa
    print("Iniciando servicio de reconocimiento de voz...")
    speech_threads = start_speech_recognition()
    print("Servicio de reconocimiento de voz iniciado correctamente.")
    
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
        opcion = input("\nSelecciona una opción (Recolectar: 1, Entrenar: 2, Señas: 3, Evaluar: 4, Salir: 5): ").strip()
        
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
            if is_trained:
                # Inicializar modo evaluación en tiempo real
                print("Modo de evaluación activado.")
                run_evaluation_mode()
            else:
                print("¡Entrena el modelo primero (Opción 2)!")
                
        elif opcion == '5':
            print("Deteniendo servicio de reconocimiento de voz...")
            stop_speech_recognition()
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida, intenta nuevamente.")
        
        # Mostrar nuevamente el menú luego de finalizar la opción seleccionada.
        print_menu()
if __name__ == "__main__":
    main()
    save_data()  # Guarda los datos recolectados
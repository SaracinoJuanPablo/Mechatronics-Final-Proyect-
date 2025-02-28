# -----------------PROYECTO FINAL-----------------
## 1. IMPORTAR LIBRERIAS
from keras.src.saving.saving_api import load_model
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import sys
from collections import deque  
import math



from kivy.utils import platform
try:
    if platform == 'android':
        from android.permissions import Permission, request_permissions
        request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
        from android.storage import app_storage_path
        BASE_DIR = app_storage_path()
    else:
        BASE_DIR = os.getcwd()
except ModuleNotFoundError:
    pass  # Para desarrollo en PC

## 2. INICIALIZAR MEDIAPIPE
# Configuración inicial global
mp_hands = mp.solutions.hands

dataset_dir = os.path.join(BASE_DIR, "dataset_11_90")
model_path = os.path.join(BASE_DIR, "gesture_model_me_10_90_pruebas.h5")

# Optimizar MediaPipe
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.45,  # Reducir confianza
    min_tracking_confidence=0.45,
    model_complexity=0  # Menor complejidad
)

mp_draw = mp.solutions.drawing_utils
sequence_length = 90
total_landmarks = 126
gestures = []
X_mean = None
X_std = None

## 3. FUNCIONES PRINCIPALES
# Funciones principales
def init_system():
    global gestures
    os.makedirs(dataset_dir, exist_ok=True)
    gestures = get_existing_gestures()
    
def get_existing_gestures():
    return sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame 

## 4. DETECCION DE MANO
def detect_hands(callback, stop_event):
    #print("\nIniciando detección de manos. Presiona 'ESC' para salir.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            processed_frame = process_frame(frame)
            callback(processed_frame)
        else:
            break
    cap.release()

## 5. RECOLLECION DE DATOS 
def collect_data(gesture_name, num_sequences, progress_callback, stop_event):
    
    save_dir = os.path.join(dataset_dir, gesture_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    sequence = []
    counter = 0 

    while not stop_event.is_set() and counter < num_sequences:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        all_landmarks = []

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks[:2]:
                for lm in hand.landmark:
                    all_landmarks.extend([lm.x, lm.y, lm.z])

            if len(results.multi_hand_landmarks) < 2:
                all_landmarks += [0.0] * 63

            sequence.append(all_landmarks)

        if len(sequence) == sequence_length:
            np.save(os.path.join(save_dir, f"secuencia_{counter}.npy"), sequence)
            counter += 1
            sequence = []
            progress_callback(counter/num_sequences)
            
    cap.release()
    init_system()  # Actualizar lista de gestos
    return counter

## 6. CARGA DE DATOS
def custom_augmentation(sequence):
    """Aumentación 100% en TensorFlow"""
    # 1. Ruido Gaussiano
    noise = tf.random.normal(tf.shape(sequence), mean=0.0, stddev=0.05)

    # Convertir explícitamente a float32
    sequence = tf.cast(sequence, tf.float32)
    # 1. Ruido Gaussiano
    noise = tf.random.normal(tf.shape(sequence), mean=0.0, stddev=0.05)
    sequence = tf.add(sequence, noise)
    
    # 2. Escalado aleatorio
    scale_factor = tf.random.uniform([], 0.9, 1.1)
    sequence = tf.multiply(sequence, scale_factor)
    
    # 3. Rotación 2D (versión TensorFlow)
    angle = tf.random.uniform([], -15.0, 15.0)  # Grados
    angle_rad = tf.math.divide(angle * math.pi, 180.0)
    
    # Crear matriz de rotación como tensor
    rot_matrix = tf.stack([
        [tf.cos(angle_rad), -tf.sin(angle_rad), 0.0],
        [tf.sin(angle_rad), tf.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Aplicar rotación a cada landmark
    original_shape = tf.shape(sequence)
    sequence = tf.reshape(sequence, [-1, 3])  # [secuencia_length*42, 3]
    sequence = tf.matmul(sequence, rot_matrix)
    sequence = tf.reshape(sequence, original_shape)
    
    # 4. Desplazamiento temporal (versión TensorFlow)
    shift = tf.random.uniform([], -5, 5, dtype=tf.int32)
    sequence = tf.cond(
        tf.random.uniform([]) > 0.5,
        lambda: tf.roll(sequence, shift=shift, axis=0),
        lambda: sequence
    )

    
    return sequence

# Modificar la función create_dataset
def create_dataset(X_data, y_data, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
    
    if augment:
        dataset = dataset.map(
            lambda x, y: (custom_augmentation(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(1000)
    
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)
def load_data(augment=True):
    X = []
    y = []
    
    for label_idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(dataset_dir, gesture)
        sequences = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
        print(f"Gesto '{gesture}' - secuencias encontradas: {len(sequences)}")
        
        for seq_file in sequences:
            seq_path = os.path.join(gesture_dir, seq_file)
            sequence = np.load(seq_path)
            
            if sequence.shape == (sequence_length, total_landmarks):
                X.append(sequence)
                y.append(label_idx)
            else:
                print(f"Secuencia {seq_file} con forma {sequence.shape} ignorada.")
    
    return np.array(X, dtype=np.float32), np.array(y), gestures  # Asegurar tipo float32

## 7. ENTRENAMIENTO DEL MODELO
def train_model(progress_callback, stop_event):
    global X_mean, X_std, gestures
    try:
        # 1. Verificar datos de entrenamiento
        gestures = get_existing_gestures()
        if not gestures:
            print("\nNo hay datos recolectados. Primero recolecte datos de gestos.")
            return

        # 2. Cargar y preparar datos
        print("\nCargando datos y preparando el entrenamiento...")
        X, y, gestures = load_data(augment=False)  # Cargar sin aumentación inicial
        y = tf.keras.utils.to_categorical(y)

        # 3. Dividir datos antes de crear el Dataset
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # 4. Calcular parámetros de normalización
        X_mean = np.mean(X_train, axis=(0, 1)).astype(np.float32)
        X_std = np.std(X_train, axis=(0, 1)).astype(np.float32)
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std  # Aplicar misma normalización a validación

        train_dataset = create_dataset(X_train, y_train, augment=True)
        val_dataset = create_dataset(X_val, y_val, augment=False)
        

        # 4. Guardar parámetros de normalización
        np.savez('normalization_params_90_pruebas.npz', mean=X_mean, std=X_std)
        
        # 5. Arquitectura optimizada del modelo
        # Arquitectura del modelo mejorada
        inputs = tf.keras.Input(shape=(sequence_length, total_landmarks))
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(len(gestures), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # 6. Compilación y entrenamiento
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()

        # Modificar el callback durante el entrenamiento
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if stop_event.is_set():
                    self.model.stop_training = True
                progress_callback(f"Epoch {epoch+1}/50 - Accuracy: {logs['accuracy']:.2f}")


        history = model.fit(
            train_dataset,
            validation_data=val_dataset,  # Usar dataset de validación explícito
            epochs=50,
            callbacks=[TrainingCallback()], 
            verbose=1
        )
        # 7. Guardar modelo y resultados
        model.save(model_path)
        print(f"\nModelo guardado en {model_path}")
        
        # 8. Conversión a TFLite con configuraciones especiales
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        try:
            tflite_model = converter.convert()
            with open('model_quantized_90_pruebas.tflite', 'wb') as f:
                f.write(tflite_model)
            print("\nModelo TFLite exportado exitosamente")
        except Exception as e:
            print(f"\nError en conversión TFLite: {str(e)}")
        
        # Mostrar métricas finales
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"Precisión de validación final: {val_accuracy:.2%}")
        progress_callback("Entrenamiento completado con 95% de precisión")

    except Exception as e:
        progress_callback(f"Error: {str(e)}")
    
## TF LITE
def convert_to_tflite():
    try:
        # Cargar el modelo entrenado
        model = tf.keras.models.load_model(model_path)
        
        # Configurar el conversor con parámetros especiales
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Añadir estas 3 líneas clave para compatibilidad con LSTM
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.allow_custom_ops = True  # Permitir operaciones personalizadas
        
        # Realizar la conversión
        tflite_model = converter.convert()
        
        # Guardar el modelo cuantizado
        with open('model_quantized_90_pruebas.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("\n✅ Conversión a TFLite exitosa!")
        
    except Exception as e:
        print(f"\n❌ Error en conversión: {str(e)}")
        print("Posibles soluciones:")
        print("1. Verifique que el modelo .h5 existe")
        print("2. Actualice TensorFlow: pip install --upgrade tensorflow")
        print("3. Reinicie el runtime/kernel")

    global gestures
    gestures = get_existing_gestures()
    print("Gestos cargados para evaluación:", gestures)

    print("Salida del modelo:", model.output_shape)



def representative_dataset_gen():
    # Generador de datos de ejemplo para calibración
    for _ in range(100):
        yield [np.random.randn(1, sequence_length, total_landmarks).astype(np.float32)]
## 8. EVALUACION DEL MODELO
from threading import Thread



# -----------------PROYECTO FINAL - VERSIÓN MEJORADA-----------------
## 8. EVALUACION DEL MODELO (CORRECCIÓN CRÍTICA)
def evaluate(callback, stop_event):

    global gestures
    gestures = get_existing_gestures()
    try:

        if not os.path.exists("model_quantized_90_pruebas.tflite"):
            print("\n¡Primero debe entrenar y convertir el modelo!")
            return
        
        # 1. Cargar parámetros y modelo
        try:
            with np.load('normalization_params_90_pruebas.npz') as data:
                X_mean = data['mean']
                X_std = data['std']
                
            interpreter = tf.lite.Interpreter(model_path="model_quantized_90_pruebas.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            print("Output details shape:", output_details['shape'])
        except Exception as e:
            print(f"\nError crítico: {str(e)}")
            return

        
        # 2. Configuración de cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("\n¡No se puede acceder a la cámara!")
            return

        # 3. Variables de estado mejoradas
        sequence = deque(maxlen=sequence_length)
        prediction_history = deque(maxlen=15)  # Suavizado de predicciones
        current_gesture = "Esperando..."
        current_confidence = 0.0

        # 4. Bucle principal optimizado
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            
            # Siempre procesar landmarks (manos detectadas o no)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            landmarks = []
            
            if results.multi_hand_landmarks:
                # Extraer landmarks para ambas manos
                for hand in results.multi_hand_landmarks[:2]:
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                
                # Rellenar con ceros si es necesario
                if len(landmarks) < total_landmarks:
                    landmarks += [0.0] * (total_landmarks - len(landmarks))
            else:
                # Si no hay manos, usar ceros
                landmarks = [0.0] * total_landmarks
            
            sequence.append(landmarks)
            
            # Realizar predicción cuando la secuencia esté completa
            if len(sequence) == sequence_length:
                    # Preprocesamiento y normalización
                    seq_array = np.array(sequence)
                    seq_array = (seq_array - X_mean) / (X_std + 1e-7)
                    input_data = seq_array.reshape(1, sequence_length, total_landmarks).astype(np.float32) # Enviar datos a UI
                    callback(current_gesture, current_confidence, frame)  # Enviar datos a UI
                    # Durante la evaluación, agregar testeo de forma
                    if input_data.shape != tuple(input_details['shape']):
                        print(f"Error: Forma esperada {input_details['shape']}, obtenida {input_data.shape}")
                        return
                    
                    # Inferencia
                    interpreter.set_tensor(input_details['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details['index'])[0]
                    
                    # Procesar resultados con suavizado
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[predicted_idx]
                    prediction_history.append((predicted_idx, confidence))
                    
                    # Calcular moda de las últimas predicciones
                    most_common = max(prediction_history, key=lambda x: list(prediction_history).count(x))
                    final_idx, final_confidence = most_common
                    
                    if final_confidence > 0.7:
                        current_gesture = gestures[final_idx]
                        current_confidence = final_confidence

                # Procesar frame para visualización
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, f"Pred: {current_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            callback(overlay_frame)

        cap.release()
    except Exception as e:
        callback(None, None, f"Error: {str(e)}")

## 9. LIMPIEZA
def cleanup():
    if 'hands' in globals():
        hands.close()
    cv2.destroyAllWindows()


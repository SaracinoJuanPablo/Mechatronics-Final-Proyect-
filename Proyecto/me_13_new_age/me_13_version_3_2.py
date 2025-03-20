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
import threading
import queue

class HandGestureRecognition:
    def __init__(self):
        # Configuración de TensorFlow para rendimiento
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU disponible para aceleración")
            except:
                print("Error al configurar GPU, usando CPU")
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configuración de directorios
        self.data_dir = "hand_gestures_data_3"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Modelo y datos de entrenamiento
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data = []
        self.labels = []
        self.model_file = "hand_gesture_nn_model_3.h5"
        self.scaler_file = "hand_gesture_scaler_3.pkl"
        self.encoder_file = "hand_gesture_encoder_3.pkl"
        
        # Estado del sistema
        self.is_trained = False
        self.is_collecting = False
        self.current_gesture = ""
        self.samples_collected = 0
        self.max_samples = 50
        
        # Control de tiempo para recolección
        self.last_sample_time = 0
        self.sample_delay = 0.05  # 50ms entre muestras
        
        # Mensajes temporales
        self.message = ""
        self.message_until = 0
        
        # Métricas
        self.metrics = {'accuracy': 0, 'val_accuracy': 0, 'training_time': 0}
        
        # Multithreading
        self.frame_queue = queue.Queue(maxsize=10)  # Cola para frames
        self.prediction = "Desconocido"
        self.confidence = 0
        self.running = True

    def extract_hand_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks_data = []
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_data.extend(landmarks)
        
        while len(landmarks_data) < 21 * 3 * 2:
            landmarks_data.append(0.0)
        
        landmarks_data = landmarks_data[:21 * 3 * 2]
        
        return landmarks_data, hands_detected
    
    def start_collection(self, gesture_name):
        self.is_collecting = True
        self.current_gesture = gesture_name
        self.samples_collected = 0
        self.set_message(f"Recolectando '{gesture_name}'...", 3)
    
    def stop_collection(self):
        self.is_collecting = False
        self.current_gesture = ""
        self.samples_collected = 0
        self.set_message("Recolección finalizada", 2)
    
    def collect_sample(self, landmarks):
        if not self.is_collecting:
            return False
        
        current_time = time.time()
        if current_time - self.last_sample_time >= self.sample_delay:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture)
            self.samples_collected += 1
            self.last_sample_time = current_time
            
            if self.samples_collected % 10 == 0:
                self.save_data()
            
            if self.samples_collected >= self.max_samples:
                self.stop_collection()
                return True
        
        return False
    
    def set_message(self, message, duration=2):
        self.message = message
        self.message_until = time.time() + duration
    
    def save_data(self):
        data = {"features": self.data, "labels": self.labels}
        with open(f"{self.data_dir}/gesture_data_3.pkl", "wb") as f:
            pickle.dump(data, f)
        self.set_message(f"Datos guardados: {len(self.data)} muestras", 1)
    
    def load_data(self):
        try:
            with open(f"{self.data_dir}/gesture_data_3.pkl", "rb") as f:
                data = pickle.load(f)
                self.data = data["features"]
                self.labels = data["labels"]
            self.set_message(f"Datos cargados: {len(self.data)} muestras", 2)
            return True
        except:
            self.set_message("No se encontraron datos previos", 2)
            return False
    
    def create_neural_network(self, input_shape, num_classes):
        """Crea una red neuronal más liviana."""
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self):
        if len(self.data) < 10:
            self.set_message("Se necesitan más datos para entrenar", 2)
            return 0
        
        X = np.array(self.data)
        y = np.array(self.labels)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        num_classes = len(set(y_encoded))
        self.set_message(f"Entrenando modelo con {num_classes} clases...", 2)
        
        self.model = self.create_neural_network(X_train.shape[1], num_classes)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        
        self.metrics = {
            'accuracy': test_accuracy,
            'val_accuracy': max(history.history['val_accuracy']),
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        }
        
        self.model.save(self.model_file)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.is_trained = True
        self.set_message(f"Modelo entrenado. Precisión: {test_accuracy:.2f}", 3)
        
        print("\n--- Informe del Modelo ---")
        print(f"Precisión en datos de prueba: {test_accuracy:.4f}")
        print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
        print("\nClasificación detallada:")
        print(self.metrics['report'])
        
        return test_accuracy
    
    def load_model(self):
        try:
            self.model = load_model(self.model_file)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.is_trained = True
            self.set_message("Modelo cargado correctamente", 2)
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            self.set_message("No se encontró un modelo guardado", 2)
            return False
    
    def predict(self, landmarks):
        if not self.is_trained or self.model is None:
            return "Desconocido", 0
        X = np.array([landmarks])
        X_scaled = self.scaler.transform(X)
        prediction_probs = self.model.predict(X_scaled, verbose=0)[0]
        prediction_idx = np.argmax(prediction_probs)
        confidence = prediction_probs[prediction_idx]
        try:
            prediction_label = self.label_encoder.inverse_transform([prediction_idx])[0]
        except:
            prediction_label = "Desconocido"
        return prediction_label if confidence >= 0.6 else "Desconocido", confidence

    def inference_thread(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                landmarks, hands_detected = self.extract_hand_landmarks(frame)
                if hands_detected:
                    self.prediction, self.confidence = self.predict(landmarks)
                else:
                    self.prediction = "No se detectan manos"
                    self.confidence = 0
                self.frame_queue.task_done()
            except queue.Empty:
                continue

    def run(self):
        self.load_data()
        self.load_model()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir resolución
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Limitar FPS

        # Iniciar hilo de inferencia
        inference_thread = threading.Thread(target=self.inference_thread)
        inference_thread.start()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Agregar frame a la cola si no está llena
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())

            # Mostrar predicción
            cv2.putText(frame, f"{self.prediction} ({self.confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar instrucciones
            cv2.putText(frame, "R: Recolectar | T: Entrenar | C: Comparar | ESC: Salir", 
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Reconocimiento de Señas", frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.running = False
            elif key == ord('r') and not self.is_collecting:
                gesture_name = input("Ingrese el nombre de la seña: ")
                if gesture_name:
                    self.start_collection(gesture_name)
            elif key == ord('t') and not self.is_collecting:
                self.train_model()
            elif key == ord('c') and self.is_trained:
                self.compare_with_previous_model()
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        self.frame_queue.put(None)  # Señalar fin al hilo
        inference_thread.join()

if __name__ == "__main__":
    hand_gesture_system = HandGestureRecognition()
    hand_gesture_system.run()

"""
OPTIMIZACION DEL MODELO 3_1
Para mejorar el rendimiento de tu sistema de evaluación en tiempo real y evitar que el video se trabe durante la grabación mientras el modelo realiza inferencias, es necesario abordar las posibles causas de la latencia y optimizar tanto el código como el flujo de trabajo. A continuación, te proporciono una explicación detallada de las causas probables, recomendaciones específicas, técnicas de optimización y un código optimizado que puedes implementar.

Posibles causas del problema
Carga computacional alta: El modelo de red neuronal puede estar consumiendo demasiados recursos (CPU/GPU) durante la inferencia, lo que afecta la captura y visualización del video.
Procesamiento síncrono: Si la captura de video y la inferencia del modelo se ejecutan en el mismo hilo, el procesamiento del modelo puede bloquear el flujo de video, causando retrasos.
Uso ineficiente de recursos: El código podría no estar aprovechando al máximo el hardware disponible o podría incluir operaciones redundantes que aumentan la carga.
Tamaño del modelo: Un modelo complejo con muchas capas o parámetros puede ser demasiado lento para ejecutarse en tiempo real, especialmente en hardware limitado.
Recomendaciones para optimizar el flujo de trabajo
1. Optimización del modelo
Reducir el tamaño del modelo: Disminuye el número de neuronas o capas para hacerlo más ligero. Por ejemplo, en tu función create_neural_network, ya tienes una arquitectura razonablemente liviana (64, 32 neuronas), pero podrías reducirla aún más si el rendimiento sigue siendo un problema (por ejemplo, a 32, 16 neuronas).
Cuantización: Convierte el modelo a un formato más eficiente (como TensorFlow Lite) para reducir el tiempo de inferencia sin afectar demasiado la precisión.
Ajuste fino: Si usas un modelo preentrenado, entrena solo las capas finales para tu tarea específica, reduciendo la complejidad.
2. Procesamiento asíncrono
Multithreading: Separa la captura de video y la inferencia en hilos diferentes. Esto permite que el video fluya sin interrupciones mientras el modelo procesa los frames en segundo plano.
Colas: Usa una cola (queue.Queue) para almacenar los frames capturados y que el hilo de inferencia los procese a su propio ritmo.
3. Optimización del código
Eliminar operaciones redundantes: Revisa funciones como predict para evitar cálculos innecesarios.
Uso eficiente de bibliotecas: Asegúrate de usar operaciones vectorizadas (por ejemplo, con NumPy) en lugar de bucles.
4. Ajustes de hardware y video
Uso de GPU: Si tienes acceso a una GPU, configúrala para acelerar la inferencia.
Reducir resolución: Baja la resolución del video (por ejemplo, a 640x480) para disminuir la cantidad de datos procesados.
Ajustar tasa de frames: Limita la tasa de frames (FPS) a 30 o menos para reducir la carga.
Técnicas específicas y código optimizado
A continuación, te proporciono un ejemplo de cómo optimizar tu sistema integrando multithreading, un modelo más liviano y ajustes en la captura de video. Este código está basado en las funciones que compartiste, pero incluye mejoras para garantizar un flujo fluido.

Explicación de las Optimizaciones Aplicadas
1. Multithreading
Objetivo: Separar la captura de video y la inferencia para que la captura sea fluida mientras el modelo procesa en segundo plano.
Implementación:
Se creó un hilo separado (inference_thread) que procesa los frames de una cola (frame_queue).
La captura de video (en run) se ejecuta en el hilo principal y agrega frames a la cola sin esperar a la inferencia.
El hilo de inferencia toma frames de la cola, realiza la predicción y actualiza las variables self.prediction y self.confidence.
2. Modelo Más Liviano
Objetivo: Reducir la complejidad computacional para acelerar la inferencia.
Implementación:
En create_neural_network, se redujo el número de neuronas de las capas ocultas de 64 y 32 a 32 y 16, disminuyendo el número total de parámetros.
Se mantuvo la regularización (l2, Dropout, BatchNormalization) para preservar la precisión, ajustando los valores de dropout a 0.2 y 0.1.
3. Ajuste de Video
Objetivo: Reducir la carga de procesamiento al disminuir la cantidad de datos capturados.
Implementación:
En run, se configuró la captura de video con:
Resolución: 640x480 píxeles.
Tasa de frames: 30 FPS.
Esto reduce significativamente el tamaño de los frames y la frecuencia de procesamiento.
4. Cola de Frames
Objetivo: Gestionar los frames capturados de manera asíncrona y evitar saturación de memoria.
Implementación:
Se usa queue.Queue con un tamaño máximo de 10 (maxsize=10) en __init__.
En run, los frames se agregan a la cola solo si no está llena (if not self.frame_queue.full()).
El hilo de inferencia (inference_thread) procesa los frames de la cola uno por uno.
5. Cuantización (Opcional)
Objetivo: Reducir aún más el tiempo de inferencia convirtiendo el modelo a un formato más eficiente.
Implementación:
Aunque no está incluida directamente en el código principal (por simplicidad), puedes agregar cuantización post-entrenamiento con TensorFlow Lite. Aquí está el proceso:
python

Contraer

Ajuste

Copiar
# Después de entrenar el modelo en train_model
converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('hand_gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
Para usar el modelo cuantizado, reemplaza la inferencia en predict con el intérprete de TensorFlow Lite:
python

Contraer

Ajuste

Copiar
interpreter = tf.lite.Interpreter(model_path='hand_gesture_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(self, landmarks):
    X = np.array([landmarks], dtype=np.float32)
    X_scaled = self.scaler.transform(X)
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    prediction_probs = interpreter.get_tensor(output_details[0]['index'])[0]
    prediction_idx = np.argmax(prediction_probs)
    confidence = prediction_probs[prediction_idx]
    prediction_label = self.label_encoder.inverse_transform([prediction_idx])[0]
    return prediction_label if confidence >= 0.6 else "Desconocido", confidence
Beneficios de las Adaptaciones
Rendimiento Mejorado: La captura de video sigue siendo fluida gracias al multithreading y la cola de frames.
Eficiencia Computacional: El modelo más liviano y el ajuste de video reducen el uso de recursos.
Escalabilidad: La cola evita problemas de memoria al limitar el número de frames en espera.
Flexibilidad: La cuantización opcional permite optimizar aún más en dispositivos con recursos limitados.
Estas modificaciones aseguran que el sistema sea más rápido y eficiente mientras mantiene su funcionalidad principal de reconocimiento de señas en tiempo real."""
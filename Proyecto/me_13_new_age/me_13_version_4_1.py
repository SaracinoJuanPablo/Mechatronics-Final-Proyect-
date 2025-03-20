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

class HandGestureRecognition:
    def __init__(self):
        """
        Inicializa el sistema de reconocimiento de señas con red neuronal.
        """
        # Configuración de TensorFlow para rendimiento
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                # Permitir crecimiento de memoria según sea necesario
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
        
        # Configuración de directorios para almacenar datos
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
        
        # Control de tiempo para la recolección continua
        self.last_sample_time = 0
        self.sample_delay = 0.05  # 50ms entre muestras (20 muestras por segundo)
        
        # Temporizador para mostrar mensajes
        self.message = ""
        self.message_until = 0
        
        # Para evaluación del modelo
        self.metrics = {
            'accuracy': 0,
            'val_accuracy': 0,
            'training_time': 0
        }
    
    def extract_hand_landmarks(self, frame):
        """
        Extrae los landmarks de las manos desde un frame de video.
        
        Args:
            frame: Imagen capturada por la cámara
            
        Returns:
            Lista de landmarks normalizados para ambas manos y booleano de detección
        """
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks_data = []
        hands_detected = False
        
        if results.multi_hand_landmarks:
            hands_detected = True
            # Extraer landmarks de hasta dos manos
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks en el frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extraer coordenadas (x,y,z) de los 21 landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_data.extend(landmarks)
        
        # Normalizar para 2 manos (si solo hay una o ninguna, rellenar con ceros)
        while len(landmarks_data) < 21 * 3 * 2:  # 21 puntos * 3 coordenadas * 2 manos
            landmarks_data.append(0.0)
        
        # Limitar a exactamente 126 características (21 puntos * 3 coordenadas * 2 manos)
        landmarks_data = landmarks_data[:21 * 3 * 2]
        
        return landmarks_data, hands_detected
    
    def start_collection(self, gesture_name):
        """
        Inicia la recolección de datos para una seña específica.
        
        Args:
            gesture_name: Nombre de la seña a recolectar
        """
        self.is_collecting = True
        self.current_gesture = gesture_name
        self.samples_collected = 0
        self.set_message(f"Mantenga la seña frente a la cámara. Recolectando '{gesture_name}'...", 3)
    
    def stop_collection(self):
        """
        Detiene la recolección de datos.
        """
        self.is_collecting = False
        self.current_gesture = ""
        self.samples_collected = 0
        self.set_message("Recolección finalizada", 2)
    
    def collect_sample(self, landmarks):
        """
        Recolecta una muestra de landmarks para la seña actual.
        
        Args:
            landmarks: Lista de landmarks extraídos
            
        Returns:
            bool: True si se ha completado la recolección, False en caso contrario
        """
        if not self.is_collecting:
            return False
        
        current_time = time.time()
        # Verificar si ha pasado suficiente tiempo desde la última muestra
        if current_time - self.last_sample_time >= self.sample_delay:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture)
            self.samples_collected += 1
            self.last_sample_time = current_time
            
            # Guardar datos periódicamente
            if self.samples_collected % 10 == 0:
                self.save_data()
            
            # Verificar si se ha completado la recolección
            if self.samples_collected >= self.max_samples:
                self.stop_collection()
                return True
        
        return False
    
    def set_message(self, message, duration=2):
        """
        Establece un mensaje para mostrar en pantalla por una duración determinada.
        
        Args:
            message: Mensaje a mostrar
            duration: Duración en segundos
        """
        self.message = message
        self.message_until = time.time() + duration
    
    def save_data(self):
        """
        Guarda los datos recolectados en disco.
        """
        data = {
            "features": self.data,
            "labels": self.labels
        }
        with open(f"{self.data_dir}/gesture_data_3.pkl", "wb") as f:
            pickle.dump(data, f)
        self.set_message(f"Datos guardados: {len(self.data)} muestras", 1)
    
    def load_data(self):
        """
        Carga los datos recolectados desde disco.
        
        Returns:
            bool: True si se cargaron los datos correctamente, False en caso contrario
        """
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
        """
        Crea una red neuronal liviana para reconocimiento de gestos.
        
        Args:
            input_shape: Dimensión de las características de entrada
            num_classes: Número de clases a predecir
            
        Returns:
            Modelo de red neuronal compilado
        """
        model = Sequential([
            # Capa de entrada con regularización para prevenir sobreajuste
            Dense(64, activation='relu', input_shape=(input_shape,), 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),  # Dropout para mejorar generalización
            
            # Capa oculta 
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Capa de salida
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar con optimizador Adam y tasa de aprendizaje reducida para estabilidad
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """
        Entrena el modelo de red neuronal con los datos recolectados.
        
        Returns:
            float: Precisión del modelo en datos de prueba
        """
        if len(self.data) < 10:
            self.set_message("Se necesitan más datos para entrenar", 2)
            return 0
        
        self.set_message("Preparando datos para entrenamiento...", 1)
        
        # Convertir a arrays NumPy
        X = np.array(self.data)
        y = np.array(self.labels)
        
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalizar datos
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Crear modelo
        num_classes = len(set(y_encoded))
        self.set_message(f"Entrenando modelo con {num_classes} clases...", 2)
        
        self.model = self.create_neural_network(X_train.shape[1], num_classes)
        
        # Callbacks para mejorar el entrenamiento
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular tiempo de entrenamiento
        training_time = time.time() - start_time
        
        # Evaluar el modelo
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        
        # Guardar métricas
        self.metrics = {
            'accuracy': test_accuracy,
            'val_accuracy': max(history.history['val_accuracy']),
            'training_time': training_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        }
        
        # Guardar el modelo y componentes
        self.model.save(self.model_file)
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.is_trained = True
        self.set_message(f"Modelo entrenado. Precisión: {test_accuracy:.2f}", 3)
        
        # Imprimir reporte detallado
        print("\n--- Informe del Modelo ---")
        print(f"Precisión en datos de prueba: {test_accuracy:.4f}")
        print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
        print("\nClasificación detallada:")
        print(self.metrics['report'])
        
        return test_accuracy
    
    def load_model(self):
        """
        Carga un modelo previamente entrenado.
        
        Returns:
            bool: True si se cargó el modelo correctamente, False en caso contrario
        """
        try:
            # Cargar modelo, scaler y codificador de etiquetas
            self.model = load_model(self.model_file)
            with open(self.scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            self.is_trained = True
            self.set_message("Modelo de red neuronal cargado correctamente", 2)
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            self.set_message("No se encontró un modelo guardado", 2)
            self.model = None
            return False
    
    def predict(self, landmarks):
        """
        Predice la seña a partir de los landmarks.
        
        Args:
            landmarks: Lista de landmarks extraídos
            
        Returns:
            str: Nombre de la seña predicha y confianza
        """
        if not self.is_trained or self.model is None:
            return None, 0
        
        # Preprocesar datos
        X = np.array([landmarks])
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        prediction_probs = self.model.predict(X_scaled, verbose=0)[0]
        prediction_idx = np.argmax(prediction_probs)
        confidence = prediction_probs[prediction_idx]
        
        # Decodificar clase
        try:
            prediction_label = self.label_encoder.inverse_transform([prediction_idx])[0]
        except:
            prediction_label = "Desconocido"
        
        # Solo devolver predicción si la confianza es suficiente
        if confidence >= 0.6:
            return prediction_label, confidence
        else:
            return "Desconocido", confidence
    
    def get_available_gestures(self):
        """
        Obtiene la lista de gestos disponibles en el conjunto de datos.
        
        Returns:
            list: Lista de nombres de gestos únicos
        """
        return list(set(self.labels))
    
    
    def run(self):
        """
        Ejecuta el sistema completo con una interfaz gráfica.
        """
        # Cargar datos y modelo si existen
        self.load_data()
        self.load_model()
        
        # Iniciar captura de video
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espejo horizontal para una experiencia más natural
            frame = cv2.flip(frame, 1)
            
            # Extraer landmarks
            landmarks, hands_detected = self.extract_hand_landmarks(frame)
            
            # Interfaz de usuario
            frame_h, frame_w, _ = frame.shape
            
            # Mostrar estado del sistema
            if self.is_collecting:
                # Barra de progreso
                progress = int((self.samples_collected / self.max_samples) * frame_w)
                cv2.rectangle(frame, (0, 0), (progress, 20), (0, 255, 0), -1)
                cv2.rectangle(frame, (0, 0), (frame_w, 20), (255, 255, 255), 1)
                
                # Texto de estado
                cv2.putText(
                    frame,
                    f"Recolectando: {self.current_gesture} ({self.samples_collected}/{self.max_samples})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Recolectar muestra si se detectan manos
                if hands_detected:
                    self.collect_sample(landmarks)
                else:
                    cv2.putText(
                        frame,
                        "¡No se detectan manos!",
                        (frame_w//2 - 100, frame_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            elif self.is_trained and hands_detected:
                prediction, confidence = self.predict(landmarks)
                conf_color = (0, 255, 0) if confidence > 0.85 else (0, 255, 255)
                
                cv2.putText(
                    frame,
                    f"Predicción: {prediction}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    conf_color,
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Confianza: {confidence:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    conf_color,
                    2
                )
            
            # Mostrar mensaje temporal
            current_time = time.time()
            if current_time < self.message_until:
                cv2.putText(
                    frame,
                    self.message,
                    (10, frame_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
            
            # Mostrar instrucciones
            cv2.putText(
                frame,
                "R: Recolectar nueva seña | T: Entrenar modelo | C: Comparar modelos | ESC: Salir",
                (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Mostrar frame
            cv2.imshow("Reconocimiento de Señas", frame)
            
            # Capturar teclas
            key = cv2.waitKey(1)
            
            # ESC para salir
            if key == 27:
                break
            
            # R para recolectar
            elif key == ord('r') and not self.is_collecting:
                gesture_name = input("Ingrese el nombre de la seña: ")
                if gesture_name:
                    self.start_collection(gesture_name)
            
            # T para entrenar
            elif key == ord('t') and not self.is_collecting:
                self.train_model()
                
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Crear y ejecutar el sistema
    hand_gesture_system = HandGestureRecognition()
    hand_gesture_system.run()